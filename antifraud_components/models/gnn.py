from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, fbeta_score, precision_recall_curve
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv
    from torch_geometric.utils import subgraph
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    _TORCH_GEO_AVAILABLE = True
except ImportError:
    _TORCH_GEO_AVAILABLE = False
ENTITY_COLS_DEFAULT = ['card_id', 'phone_hash', 'device_id']
MAX_GROUP_SIZE = 50
TARGET_COL = 'resolution_fraud'
BETA = 2.0
HIDDEN_DIM = 64
DROPOUT = 0.3
LEARNING_RATE = 0.001
EPOCHS = 50

def _check_pyg():
    if not _TORCH_GEO_AVAILABLE:
        raise ImportError('torch and torch_geometric are required for GNN. Install: pip install torch torch_geometric')

def build_edge_list(df: pd.DataFrame, entity_cols: list[str], id_col: str='order_id', max_group: int=MAX_GROUP_SIZE) -> 'tuple[dict, np.ndarray, np.ndarray]':
    import gc as _gc
    if not entity_cols:
        return ({}, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    order_to_idx = pd.Series(df.index, index=df[id_col].values)
    edge_indices_by_type: dict = {}
    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    for col in entity_cols:
        if col not in df.columns:
            continue
        valid = df[[id_col, col]].copy()
        valid[col] = valid[col].astype(str).str.strip()
        valid = valid[~valid[col].isin(['', 'nan', 'None', 'NaN', 'null'])]
        grp_size = valid.groupby(col)[id_col].transform('count')
        valid = valid[(grp_size >= 2) & (grp_size <= max_group)]
        if valid.empty:
            del valid
            continue
        merged = valid.merge(valid, on=col, suffixes=('_a', '_b'))
        del valid
        merged = merged[merged[f'{id_col}_a'] < merged[f'{id_col}_b']]
        src = order_to_idx[merged[f'{id_col}_a'].values].values.astype(np.int64)
        dst = order_to_idx[merged[f'{id_col}_b'].values].values.astype(np.int64)
        del merged
        _gc.collect()
        edge_indices_by_type[col] = torch.tensor(np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])]), dtype=torch.long)
        src_parts.append(src)
        dst_parts.append(dst)
        print(f'  {col}: {len(src):,} edges')
        del src, dst
    if not src_parts:
        return ({}, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    src_all = np.concatenate(src_parts)
    dst_all = np.concatenate(dst_parts)
    del src_parts, dst_parts
    _gc.collect()
    pairs = np.unique(np.stack([src_all, dst_all], axis=1), axis=0)
    del src_all, dst_all
    print(f'\nTotal edges (deduped): {len(pairs):,}')
    return (edge_indices_by_type, pairs[:, 0], pairs[:, 1])

def _str_oid(x) -> str:
    return str(x).strip()

def build_graph_data(df: pd.DataFrame, entity_cols: list[str] | None=None, cat_cols: list[str] | None=None, exclude_cols: set[str] | None=None, feature_cols: list[str] | None=None, id_col: str | None='order_id', target_col: str=TARGET_COL, train_order_ids: np.ndarray | None=None, val_order_ids: np.ndarray | None=None, test_order_ids: np.ndarray | None=None, y_train_labels: 'np.ndarray | None'=None, y_val_labels: 'np.ndarray | None'=None, y_test_labels: 'np.ndarray | None'=None) -> tuple['Data', StandardScaler, list[str], np.ndarray]:
    _check_pyg()
    entity_cols = entity_cols or [c for c in ENTITY_COLS_DEFAULT if c in df.columns]
    cat_cols = cat_cols or [c for c in ['merchant_id', 'split_type', 'bank_name'] if c in df.columns]
    target_and_meta = {target_col, 'resolution_fraud', 'resolution_negative', 'resolution_no_fraud', 'resolution_lvl_1', 'resolution_lvl_2', 'fraud_flg', 'ead7_2', 'ead30_2', 'due_date_flg7_2', 'due_date_flg30_2', 'npv_37_day_with_market_compensation', 'prod_37_day_npv_pred'}
    df_feat = df.sort_values('order_creation_dttm').reset_index(drop=True).copy()
    if id_col is None or id_col not in df_feat.columns:
        id_col = '__gnn_internal_id__'
        df_feat[id_col] = np.arange(len(df_feat), dtype=np.int64)
    non_feature = {id_col, 'order_creation_dttm', 'max_iso_eventtime_str'} | (exclude_cols or set())
    _use_tabular_pre = train_order_ids is not None and val_order_ids is not None and (test_order_ids is not None) and (len(train_order_ids) > 0) and (len(val_order_ids) > 0) and (len(test_order_ids) > 0)
    if _use_tabular_pre:
        _tr_set_pre = {_str_oid(o) for o in np.asarray(train_order_ids).ravel()}
        _oids_pre = df_feat[id_col].map(_str_oid).to_numpy()
        _train_row_mask = np.array([o in _tr_set_pre for o in _oids_pre], dtype=bool)
    else:
        _labeled_pre = df_feat[df_feat[target_col].notna()].index.values
        _train_end_pre = int(len(_labeled_pre) * 0.65)
        _train_row_mask = np.zeros(len(df_feat), dtype=bool)
        _train_row_mask[_labeled_pre[:_train_end_pre]] = True
    for col in cat_cols:
        le = LabelEncoder()
        train_vals = df_feat.loc[_train_row_mask, col].astype(str).fillna('unknown')
        all_classes = sorted(set(train_vals.tolist()) | {'unknown'})
        le.fit(all_classes)
        known = set(le.classes_)
        safe = df_feat[col].astype(str).fillna('unknown').map(lambda x: x if x in known else 'unknown')
        df_feat[f'{col}_enc'] = le.transform(safe)
    if feature_cols is not None:
        _requested = [c for c in feature_cols if c != '__node_degree__']
        _missing = [c for c in _requested if c not in df_feat.columns]
        if _missing:
            print(f'  GNN: {len(_missing)} признаков из чекпойнта отсутствуют в df — авто-определение признаков')
            feature_cols = None
        else:
            feature_cols = _requested
            print(f'  GNN: признаки зафиксированы по чекпойнту ({len(feature_cols)} шт.)')
    if feature_cols is None:
        numeric_cols = df_feat.select_dtypes(include='number').columns.tolist()
        cat_enc_cols = [f'{c}_enc' for c in cat_cols]
        feature_cols = [c for c in numeric_cols + cat_enc_cols if c not in target_and_meta and c not in non_feature and (c not in entity_cols)]
    print(f'\n=== GNN feature_cols ({len(feature_cols)} признаков) ===')
    for i, fc in enumerate(feature_cols):
        print(f'  {i + 1:3d}. {fc}')
    print('=' * 40)
    for col in feature_cols:
        train_median = df_feat.loc[_train_row_mask, col].median()
        df_feat[col] = df_feat[col].fillna(train_median)
    scaler = StandardScaler()
    scaler.fit(df_feat.loc[_train_row_mask, feature_cols].values.astype(np.float32))
    feature_matrix = scaler.transform(df_feat[feature_cols].values.astype(np.float32))
    x = torch.tensor(feature_matrix, dtype=torch.float)
    del feature_matrix
    n_nodes = len(df_feat)
    y_arr = np.full(n_nodes, -1, dtype=np.int64)
    labeled_idx = df_feat[df_feat[target_col].notna()].index.values
    y_arr[labeled_idx] = df_feat.loc[labeled_idx, target_col].astype(int).values
    y = torch.tensor(y_arr, dtype=torch.long)
    entity_cols_avail = [c for c in entity_cols if c in df_feat.columns]
    import gc as _gc
    edge_indices_by_type, src_dedup, dst_dedup = build_edge_list(df_feat, entity_cols_avail, id_col=id_col)
    if len(src_dedup) > 0:
        edge_index = torch.tensor(np.stack([np.concatenate([src_dedup, dst_dedup]), np.concatenate([dst_dedup, src_dedup])]), dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        ec_hint = entity_cols_avail if entity_cols_avail else 'ни одной из ENTITY_COLS не найдено в df'
        print(f'  GNN: рёбер нет — ни у одного заказа не совпало значение entity с другим заказом (после фильтра пустых и групп >{MAX_GROUP_SIZE}). Колонки для рёбер: {ec_hint}. Нужны повторяющиеся card_id / device_id / … (см. data/df_gnn.parquet).')
    del src_dedup, dst_dedup
    _gc.collect()

    def idx_to_mask(idx_arr: np.ndarray) -> 'torch.Tensor':
        m = torch.zeros(n_nodes, dtype=torch.bool)
        if len(idx_arr):
            m[torch.as_tensor(idx_arr, dtype=torch.long)] = True
        return m
    use_tabular = train_order_ids is not None and val_order_ids is not None and (test_order_ids is not None) and (len(train_order_ids) > 0) and (len(val_order_ids) > 0) and (len(test_order_ids) > 0)
    if use_tabular:
        train_set = {_str_oid(x) for x in np.asarray(train_order_ids).ravel()}
        val_set = {_str_oid(x) for x in np.asarray(val_order_ids).ravel()}
        test_set = {_str_oid(x) for x in np.asarray(test_order_ids).ravel()}
        ov = train_set & val_set | train_set & test_set | val_set & test_set
        if ov:
            raise ValueError(f'Пересечение train/val/test по order_id: {len(ov)} примеров')
        tr_m = torch.zeros(n_nodes, dtype=torch.bool)
        va_m = torch.zeros(n_nodes, dtype=torch.bool)
        te_m = torch.zeros(n_nodes, dtype=torch.bool)
        oids = df_feat[id_col].map(_str_oid).to_numpy()
        _oids_s = pd.Series(oids)
        _tr_idx = np.where(_oids_s.isin(train_set))[0]
        _va_idx = np.where(_oids_s.isin(val_set))[0]
        _te_idx = np.where(_oids_s.isin(test_set))[0]
        del _oids_s
        if len(_tr_idx):
            tr_m[torch.as_tensor(_tr_idx, dtype=torch.long)] = True
        if len(_va_idx):
            va_m[torch.as_tensor(_va_idx, dtype=torch.long)] = True
        if len(_te_idx):
            te_m[torch.as_tensor(_te_idx, dtype=torch.long)] = True
        del _tr_idx, _va_idx, _te_idx
        labeled = y >= 0
        train_mask = tr_m & labeled
        val_mask = va_m & labeled
        test_mask = te_m & labeled
        n_orphan = int(n_nodes - (tr_m | va_m | te_m).sum().item())
        if n_orphan > 0:
            print(f'  GNN: узлов графа вне табличного train/val/test (по order_id): {n_orphan:,} (сообщения проходят, loss/метрики только по размеченным маскам).')
        if y_train_labels is not None and y_val_labels is not None and (y_test_labels is not None):
            oid_to_label: dict = {}
            for arr_oids, arr_y in [(train_order_ids, np.asarray(y_train_labels, dtype=float).ravel()), (val_order_ids, np.asarray(y_val_labels, dtype=float).ravel()), (test_order_ids, np.asarray(y_test_labels, dtype=float).ravel())]:
                for o, lbl in zip(np.asarray(arr_oids).ravel(), arr_y):
                    if not np.isnan(lbl):
                        oid_to_label[_str_oid(o)] = int(lbl)
            label_mapped = pd.Series(oids).map(oid_to_label)
            upd = label_mapped.notna().values
            if upd.sum() > 0:
                y_arr[upd] = label_mapped[upd].astype(np.int64).values
                y = torch.tensor(y_arr, dtype=torch.long)
                labeled_new = y >= 0
                train_mask = tr_m & labeled_new
                val_mask = va_m & labeled_new
                test_mask = te_m & labeled_new
                n_fraud_tr = int((y[train_mask] == 1).sum().item())
                n_tr = int(train_mask.sum().item())
                print(f'  GNN: y переопределён из табличного сплита ({upd.sum():,} узлов; fraud rate в train = {n_fraud_tr}/{n_tr} = {n_fraud_tr / max(n_tr, 1):.2%})')
        print(f'  GNN: маски = табличный сплит DataPreparer → train={train_mask.sum().item():,}  val={val_mask.sum().item():,}  test={test_mask.sum().item():,} узлов с меткой')
    else:
        n_labeled = len(labeled_idx)
        train_end = int(n_labeled * 0.65)
        val_end = int(n_labeled * 0.8)
        train_idx = labeled_idx[:train_end]
        val_idx = labeled_idx[train_end:val_end]
        test_idx = labeled_idx[val_end:]
        train_mask = idx_to_mask(train_idx)
        val_mask = idx_to_mask(val_idx)
        test_mask = idx_to_mask(test_idx)
        print('  GNN: маски = внутренний сплит 65/15/20 по размеченным узлам (legacy; для сравнения с GBM передайте train/val/test order_id из DataPreparer).')
    if edge_index.size(1) > 0:
        from torch_geometric.utils import degree as _pyg_degree
        deg = _pyg_degree(edge_index[0], num_nodes=n_nodes, dtype=torch.float).unsqueeze(1)
        deg = torch.log1p(deg)
        x = torch.cat([x, deg], dim=1)
        feature_cols = list(feature_cols) + ['__node_degree__']
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    node_order_ids = df_feat[id_col].to_numpy()
    del df_feat
    _gc.collect()
    data.order_ids_np = np.asarray(node_order_ids)
    data.edges_typed_df = None
    data.edge_indices_by_type = edge_indices_by_type
    print(data)
    return (data, scaler, feature_cols, node_order_ids)

class FraudGraphSAGE(torch.nn.Module if _TORCH_GEO_AVAILABLE else object):

    def __init__(self, in_channels: int, hidden: int=HIDDEN_DIM, dropout: float=DROPOUT) -> None:
        if not _TORCH_GEO_AVAILABLE:
            raise ImportError('torch and torch_geometric are required')
        super().__init__()
        self.dropout = dropout
        out_dim = hidden // 2
        self.conv1 = SAGEConv(in_channels, hidden)
        self.bn1 = torch.nn.BatchNorm1d(hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.bn2 = torch.nn.BatchNorm1d(hidden)
        self.conv3 = SAGEConv(hidden, out_dim)
        self.bn3 = torch.nn.BatchNorm1d(out_dim)
        self.skip = torch.nn.Linear(in_channels, out_dim, bias=False)
        self.head = torch.nn.Linear(out_dim, 2)

    def forward(self, x, edge_index):
        residual = self.skip(x)
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.bn3(self.conv3(h, edge_index)))
        return self.head(h + residual)

    def predict_proba(self, x, edge_index) -> 'torch.Tensor':
        return torch.softmax(self.forward(x, edge_index), dim=-1)[:, 1]

def _compute_fbeta(y_true, y_prob, beta: float=BETA) -> tuple[float, float]:
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f_beta = (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec + 1e-09)
    best_idx = int(np.argmax(f_beta))
    threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return (float(f_beta[best_idx]), float(threshold))

def _fbeta_on_test_with_val_threshold(y_val: np.ndarray, p_val: np.ndarray, y_test: np.ndarray, p_test: np.ndarray, *, beta: float=BETA) -> tuple[float, float]:
    best_score = -1.0
    best_th = 0.5
    for th in np.arange(0.01, 0.99, 0.01):
        y_pred_v = (p_val >= th).astype(int)
        score = fbeta_score(y_val, y_pred_v, beta=beta, zero_division=0)
        if score > best_score:
            best_score = score
            best_th = float(th)
    y_pred_te = (p_test >= best_th).astype(int)
    fb_te = fbeta_score(y_test, y_pred_te, beta=beta, zero_division=0)
    return (best_th, fb_te)

def _write_gnn_meta(output_dir: Path, *, in_channels: int, hidden: int, dropout: float, feature_cols: list[str] | None, best_epoch: int, best_val_fbeta: float, split_mode: str | None=None) -> None:
    meta = {'in_channels': in_channels, 'hidden': hidden, 'dropout': dropout, 'feature_cols': list(feature_cols or []), 'best_epoch': best_epoch, 'best_val_fbeta': best_val_fbeta}
    if split_mode is not None:
        meta['split_mode'] = split_mode
    (output_dir / 'gnn_meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')

def _load_gnn_history(output_dir: Path) -> dict:
    hist_path = output_dir / 'gnn_history.json'
    if not hist_path.exists():
        return {'train_loss': [], 'val_fbeta': [], 'val_prauc': [], 'skipped': True}
    try:
        return json.loads(hist_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {'train_loss': [], 'val_fbeta': [], 'val_prauc': [], 'skipped': True}

def try_load_gnn_from_checkpoint(output_dir: str | Path, in_channels: int, hidden: int=HIDDEN_DIM, dropout: float=DROPOUT, feature_cols: list[str] | None=None, device: Optional['torch.device']=None, split_mode: str | None=None) -> tuple['FraudGraphSAGE', dict] | None:
    _check_pyg()
    out = Path(output_dir)
    meta_path = out / 'gnn_meta.json'
    w_path = out / 'best_model.pt'
    if not meta_path.is_file() or not w_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if meta.get('in_channels') != in_channels:
        return None
    if meta.get('hidden') != hidden or float(meta.get('dropout', DROPOUT)) != float(dropout):
        return None
    saved_cols = meta.get('feature_cols') or []
    if feature_cols is not None and saved_cols != list(feature_cols):
        return None
    if split_mode is not None and meta.get('split_mode') != split_mode:
        return None
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FraudGraphSAGE(in_channels=in_channels, hidden=hidden, dropout=dropout).to(device)
    state = torch.load(w_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    history = _load_gnn_history(out)
    history['loaded_from_checkpoint'] = True
    history['checkpoint_path'] = str(w_path.resolve())
    return (model, history)

def train_gnn(data: 'Data', in_channels: int, hidden: int=HIDDEN_DIM, dropout: float=DROPOUT, lr: float=LEARNING_RATE, epochs: int=EPOCHS, early_stopping_patience: int=0, output_dir: str | Path | None='outputs/gnn', device: Optional['torch.device']=None, feature_cols: list[str] | None=None, skip_if_checkpoint: bool=False, split_mode: str | None='tabular') -> tuple['FraudGraphSAGE', dict]:
    _check_pyg()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    if skip_if_checkpoint and output_dir is not None:
        loaded = try_load_gnn_from_checkpoint(output_dir, in_channels=in_channels, hidden=hidden, dropout=dropout, feature_cols=feature_cols, device=device, split_mode=split_mode)
        if loaded is not None:
            model, history = loaded
            print(f"  GNN: загружен чекпойнт (обучение пропущено): {Path(output_dir) / 'best_model.pt'}")
            return (model, history)
    model = FraudGraphSAGE(in_channels=in_channels, hidden=hidden, dropout=dropout).to(device)
    data = data.to(device)
    train_labels = data.y[data.train_mask].cpu().numpy()
    n_neg = (train_labels == 0).sum()
    n_pos = (train_labels == 1).sum()
    pos_weight = n_neg / max(n_pos, 1)
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float).to(device)

    def focal_loss(logits, targets, gamma=2.0):
        ce = torch.nn.functional.cross_entropy(logits, targets, weight=class_weights, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** gamma * ce).mean()
    criterion = focal_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    history = {'train_loss': [], 'val_fbeta': [], 'val_prauc': []}
    best_val_fbeta = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask].long())
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            probs = model.predict_proba(data.x, data.edge_index).cpu().numpy()
        val_y = data.y[data.val_mask].cpu().numpy()
        val_prob = probs[data.val_mask.cpu().numpy()]
        val_fbeta, _ = _compute_fbeta(val_y, val_prob)
        val_prauc = average_precision_score(val_y, val_prob)
        scheduler.step(1 - val_fbeta)
        history['train_loss'].append(float(loss.item()))
        history['val_fbeta'].append(val_fbeta)
        history['val_prauc'].append(val_prauc)
        if val_fbeta > best_val_fbeta:
            best_val_fbeta = val_fbeta
            best_epoch = epoch
            epochs_no_improve = 0
            if output_dir is not None:
                out_p = Path(output_dir)
                torch.save(model.state_dict(), out_p / 'best_model.pt')
                _write_gnn_meta(out_p, in_channels=in_channels, hidden=hidden, dropout=dropout, feature_cols=feature_cols, best_epoch=best_epoch, best_val_fbeta=best_val_fbeta, split_mode=split_mode)
        else:
            epochs_no_improve += 1
        if True:
            print(f'Epoch {epoch:3d} | loss={loss.item():.4f} | val F_β={val_fbeta:.4f} | val PR-AUC={val_prauc:.4f}')
        if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
            print(f'  Early stopping на эпохе {epoch} (нет улучшения {early_stopping_patience} эпох подряд)')
            break
    print(f'\nBest epoch: {best_epoch}  F_β={best_val_fbeta:.4f}')
    if output_dir is not None:
        best_path = Path(output_dir) / 'best_model.pt'
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        out_p = Path(output_dir)
        _write_gnn_meta(out_p, in_channels=in_channels, hidden=hidden, dropout=dropout, feature_cols=feature_cols, best_epoch=best_epoch, best_val_fbeta=best_val_fbeta, split_mode=split_mode)
        (out_p / 'gnn_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    model.eval()
    return (model, history)

def _sample_nodes_connected_preview(ei_np_full: np.ndarray, num_nodes: int, max_nodes: int, rng: np.random.Generator, *, max_edges_scan: int=120000) -> np.ndarray | None:
    import networkx as nx
    n_e = int(ei_np_full.shape[1])
    if n_e == 0:
        return None
    if n_e > max_edges_scan:
        idx = rng.choice(n_e, size=max_edges_scan, replace=False)
        ei_part = ei_np_full[:, idx]
    else:
        ei_part = ei_np_full
    G = nx.Graph()
    for i in range(ei_part.shape[1]):
        u, v = (int(ei_part[0, i]), int(ei_part[1, i]))
        if u != v:
            G.add_edge(u, v)
    if G.number_of_edges() == 0:
        return None
    ccs = list(nx.connected_components(G))
    core = list(max(ccs, key=len))
    if len(core) <= max_nodes:
        return np.sort(np.asarray(core, dtype=np.int64))
    start = int(rng.choice(core))
    sg = G.subgraph(core)
    order: list[int] = [start]
    seen: set[int] = {start}
    frontier: list[int] = [start]
    while len(order) < max_nodes and frontier:
        nxt: list[int] = []
        for x in frontier:
            for nb in sg.neighbors(x):
                if nb not in seen and len(order) < max_nodes:
                    seen.add(nb)
                    order.append(nb)
                    nxt.append(nb)
        frontier = nxt
    return np.sort(np.asarray(order[:max_nodes], dtype=np.int64))

def plot_gnn_subgraph_preview(data: 'Data', output_path: str | Path='figures/gnn_graph_preview.png', max_nodes: int=400, seed: int=42) -> Path | None:
    _check_pyg()
    import matplotlib.pyplot as plt
    import networkx as nx
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ei = data.edge_index.cpu()
    if ei.numel() == 0:
        fig, ax = plt.subplots(figsize=(7, 2.2))
        ax.text(0.5, 0.5, 'Нет рёбер — граф без связей по entity-колонкам', ha='center', va='center', fontsize=11)
        ax.axis('off')
        fig.savefig(output_path, dpi=140, bbox_inches='tight')
        plt.close(fig)
        return output_path
    rng = np.random.default_rng(seed)
    ei_np_full = ei.numpy()
    n_e = int(ei_np_full.shape[1])
    perm = rng.permutation(n_e)
    nodes_set: set[int] = set()
    for j in perm:
        u, v = (int(ei_np_full[0, j]), int(ei_np_full[1, j]))
        if u == v:
            continue
        nu = 0 if u in nodes_set else 1
        nv = 0 if v in nodes_set else 1
        if len(nodes_set) + nu + nv <= max_nodes:
            nodes_set.add(u)
            nodes_set.add(v)
        if len(nodes_set) >= max_nodes:
            break
    if not nodes_set:
        nodes = torch.unique(ei.flatten())
        n = int(nodes.numel())
        if n > max_nodes:
            pick = rng.choice(nodes.numpy(), size=max_nodes, replace=False)
            subset = torch.tensor(np.sort(pick), dtype=torch.long)
        else:
            subset = nodes
    else:
        subset = torch.tensor(sorted(nodes_set), dtype=torch.long)
    ei_cpu = data.edge_index.cpu().contiguous()

    def _induced_edges(sub: torch.Tensor) -> tuple[torch.Tensor, int]:
        e_sub, _ = subgraph(sub, ei_cpu, relabel_nodes=True, num_nodes=data.num_nodes)
        return (e_sub, int(e_sub.shape[1]))
    edge_index_sub, n_e_sub = _induced_edges(subset)
    if n_e_sub == 0 and n_e > 0:
        alt = _sample_nodes_connected_preview(ei_np_full, int(data.num_nodes), max_nodes, rng)
        if alt is not None and len(alt) > 0:
            subset = torch.tensor(alt, dtype=torch.long)
            edge_index_sub, n_e_sub = _induced_edges(subset)
    if n_e_sub == 0 and n_e > 0:
        for j in range(n_e):
            u, v = (int(ei_np_full[0, j]), int(ei_np_full[1, j]))
            if u != v:
                subset = torch.tensor([u, v], dtype=torch.long)
                edge_index_sub, _ = _induced_edges(subset)
                break
    n_sub = int(subset.numel())
    y_full = data.y.cpu().numpy()
    colors: list[str] = []
    for global_i in subset.tolist():
        lab = int(y_full[global_i])
        if lab == 1:
            colors.append('#c0392b')
        elif lab == 0:
            colors.append('#2980b9')
        else:
            colors.append('#95a5a6')
    G = nx.Graph()
    G.add_nodes_from(range(n_sub))
    ei_np = edge_index_sub.cpu().numpy()
    for i in range(ei_np.shape[1]):
        u, v = (int(ei_np[0, i]), int(ei_np[1, i]))
        if u != v:
            G.add_edge(u, v)
    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=seed, k=0.18 / max(np.sqrt(n_sub), 1))
    n_ge = G.number_of_edges()
    edge_alpha = 0.55 if n_ge < 5000 else 0.35
    edge_w = max(1.2, 1.0 if n_ge < 3000 else 0.8)
    nx.draw_networkx_edges(G, pos, alpha=edge_alpha, width=edge_w, ax=ax, edge_color='#34495e', arrows=False)
    nx.draw_networkx_nodes(G, pos, node_size=42, node_color=colors, alpha=0.95, linewidths=0, ax=ax)
    ax.set_axis_off()
    ax.set_title(f'Фрагмент графа заказов (узлов в сэмпле: {n_sub}, рёбер: {G.number_of_edges()})\nкрасный — фрод, синий — не фрод, серый — без метки', fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return output_path

def gnn_predict(model: 'FraudGraphSAGE', data: 'Data', device: Optional['torch.device']=None) -> np.ndarray:
    _check_pyg()
    if device is None:
        device = torch.device('cpu')
    model = model.to(device)
    data = data.to(device)
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(data.x, data.edge_index).cpu().numpy()
    return probs

def evaluate_gnn(model: 'FraudGraphSAGE', data: 'Data', mask_name: str='test_mask', *, threshold_on_test: bool=False) -> dict:
    _check_pyg()
    probs = gnn_predict(model, data)
    val_m = data.val_mask.cpu().numpy()
    mask = getattr(data, mask_name).cpu().numpy()
    y_val = data.y.cpu().numpy()[val_m]
    p_val = probs[val_m]
    y_true = data.y.cpu().numpy()[mask]
    y_prob = probs[mask]
    if threshold_on_test:
        fbeta, threshold = _compute_fbeta(y_true, y_prob)
    else:
        threshold, fbeta = _fbeta_on_test_with_val_threshold(y_val, p_val, y_true, y_prob, beta=BETA)
    return {'F_beta': fbeta, 'threshold': threshold, 'PR-AUC': average_precision_score(y_true, y_prob), 'ROC-AUC': roc_auc_score(y_true, y_prob)}

def plot_gnn_interactive(data: 'Data', order_ids: Optional[np.ndarray]=None, probs: Optional[np.ndarray]=None, node_meta_df: Optional[pd.DataFrame]=None, max_nodes: int=300, seed: int=42, output_path: 'str | Path | None'='outputs/gnn/gnn_interactive.html', show_in_notebook: bool=True) -> Optional[Path]:
    _check_pyg()
    try:
        import plotly.graph_objects as go
        import networkx as nx
    except ImportError as exc:
        raise ImportError('plotly и networkx обязательны для интерактивной визуализации') from exc
    rng = np.random.default_rng(seed)
    n_nodes = int(data.num_nodes)
    oids = order_ids if order_ids is not None else getattr(data, 'order_ids_np', None)
    oids = np.asarray(oids).ravel() if oids is not None else np.arange(n_nodes, dtype=np.int64)
    y_arr = data.y.cpu().numpy()
    pr = np.asarray(probs).ravel() if probs is not None else None
    train_m = data.train_mask.cpu().numpy()
    val_m = data.val_mask.cpu().numpy()
    test_m = data.test_mask.cpu().numpy()
    meta_lookup: dict = {}
    if node_meta_df is not None and 'order_id' in node_meta_df.columns:
        for _, row in node_meta_df.iterrows():
            meta_lookup[str(row['order_id'])] = row
    ei_np = data.edge_index.cpu().numpy()
    fraud_idx = set((int(i) for i in np.where(y_arr == 1)[0]))
    n_e = int(ei_np.shape[1])
    adj: dict[int, set[int]] = {}
    for k in range(n_e):
        u, v = (int(ei_np[0, k]), int(ei_np[1, k]))
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    selected: set[int] = set()
    queue = list(fraud_idx)
    rng.shuffle(queue)
    for node in queue:
        if len(selected) >= max_nodes:
            break
        selected.add(node)
        for nb in adj.get(node, set()):
            if len(selected) >= max_nodes:
                break
            selected.add(nb)
    if len(selected) < max_nodes:
        pool = [i for i in range(n_nodes) if i not in selected]
        rng.shuffle(pool)
        for i in pool:
            if len(selected) >= max_nodes:
                break
            if adj.get(i, set()) & selected:
                selected.add(i)
    if len(selected) < max_nodes:
        pool = [i for i in range(n_nodes) if i not in selected]
        rng.shuffle(pool)
        for i in pool[:max_nodes - len(selected)]:
            selected.add(i)
    sub_nodes = sorted(selected)
    sub_set = set(sub_nodes)
    global_to_local = {g: l for l, g in enumerate(sub_nodes)}
    G = nx.Graph()
    G.add_nodes_from(range(len(sub_nodes)))
    edges_typed_df: Optional[pd.DataFrame] = getattr(data, 'edges_typed_df', None)
    edge_records: list[tuple[int, int, str]] = []
    if edges_typed_df is not None and (not edges_typed_df.empty):
        for _, row in edges_typed_df.iterrows():
            s, d = (int(row['src_idx']), int(row['dst_idx']))
            if s in sub_set and d in sub_set:
                ls, ld = (global_to_local[s], global_to_local[d])
                et = str(row.get('entity_type', 'unknown'))
                edge_records.append((ls, ld, et))
                G.add_edge(ls, ld)
    else:
        for k in range(n_e):
            s, d = (int(ei_np[0, k]), int(ei_np[1, k]))
            if s in sub_set and d in sub_set and (s < d):
                ls, ld = (global_to_local[s], global_to_local[d])
                edge_records.append((ls, ld, 'linked'))
                G.add_edge(ls, ld)
    pos = nx.spring_layout(G, seed=seed, k=0.3 / max(np.sqrt(len(sub_nodes)), 1))
    entity_types = sorted(set((et for _, _, et in edge_records)))
    _palette = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22', '#34495e']
    et_color = {et: _palette[i % len(_palette)] for i, et in enumerate(entity_types)}
    traces: list = []
    for et in entity_types:
        ex, ey, mx, my = ([], [], [], [])
        for ls, ld, edge_et in edge_records:
            if edge_et != et:
                continue
            x0, y0 = pos[ls]
            x1, y1 = pos[ld]
            ex += [x0, x1, None]
            ey += [y0, y1, None]
            mx.append((x0 + x1) / 2)
            my.append((y0 + y1) / 2)
        if not ex:
            continue
        traces.append(go.Scatter(x=ex, y=ey, mode='lines', line=dict(color=et_color[et], width=1.2), hoverinfo='none', showlegend=True, name=f'ребро: {et}'))
        traces.append(go.Scatter(x=mx, y=my, mode='markers', marker=dict(size=6, color=et_color[et], opacity=0.0), hovertemplate=f'<b>entity: {et}</b><extra></extra>', showlegend=False, name=f'_mid_{et}'))
    node_x, node_y = ([], [])
    node_colors, node_sizes, node_symbols = ([], [], [])
    hover_texts = []
    node_label_names = []
    _NODE_COLOR = {'fraud': '#e74c3c', 'non-fraud': '#2980b9', 'unlabeled': '#95a5a6'}
    for local_i, global_i in enumerate(sub_nodes):
        x, y = pos[local_i]
        node_x.append(x)
        node_y.append(y)
        label_int = int(y_arr[global_i])
        if label_int == 1:
            label_str = 'fraud'
        elif label_int == 0:
            label_str = 'non-fraud'
        else:
            label_str = 'unlabeled'
        if train_m[global_i]:
            split = 'train'
        elif val_m[global_i]:
            split = 'val'
        elif test_m[global_i]:
            split = 'test'
        else:
            split = '—'
        fraud_prob_str = f'{pr[global_i]:.3f}' if pr is not None else 'n/a'
        oid = str(oids[global_i])
        lines = [f'<b>order_id: {oid}</b>', f'label: {label_str}  |  split: {split}', f'P(fraud): {fraud_prob_str}']
        meta = meta_lookup.get(oid)
        if meta is not None:
            for col in ['order_loan', 'card_age', 'phone_age', 'puid_age', 'glue_size', 'glue_ead']:
                if col in meta.index and pd.notna(meta[col]):
                    lines.append(f'{col}: {meta[col]}')
        hover_texts.append('<br>'.join(lines))
        node_colors.append(_NODE_COLOR[label_str])
        node_label_names.append(label_str)
        if label_int == 1:
            node_sizes.append(14)
            node_symbols.append('diamond')
        elif label_int == 0:
            node_sizes.append(9)
            node_symbols.append('circle')
        else:
            node_sizes.append(7)
            node_symbols.append('circle-open')
    traces.append(go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(size=node_sizes, color=node_colors, symbol=node_symbols, line=dict(width=0.8, color='#2c3e50')), hovertemplate='%{text}<extra></extra>', text=hover_texts, showlegend=False, name='заказ'))
    for lbl, col, sym in [('fraud', '#e74c3c', 'diamond'), ('non-fraud', '#2980b9', 'circle'), ('unlabeled', '#95a5a6', 'circle-open')]:
        traces.append(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color=col, symbol=sym), showlegend=True, name=f'узел: {lbl}'))
    fig = go.Figure(data=traces, layout=go.Layout(title=dict(text=f'Граф заказов — {len(sub_nodes)} узлов, {len(edge_records)} рёбер (из {n_nodes:,} всего)', font=dict(size=14)), showlegend=True, hovermode='closest', margin=dict(l=20, r=20, t=50, b=20), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), plot_bgcolor='white', paper_bgcolor='white', legend=dict(itemsizing='constant')))
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out), include_plotlyjs='cdn')
        print(f'  Интерактивный граф сохранён: {out}')
    if show_in_notebook:
        try:
            from IPython import get_ipython
            from IPython.display import display
            if get_ipython() is not None:
                fig.show()
        except Exception:
            pass
    return Path(output_path) if output_path is not None else None
