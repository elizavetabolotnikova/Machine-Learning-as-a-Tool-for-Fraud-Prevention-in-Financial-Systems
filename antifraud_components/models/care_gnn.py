from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
    from sklearn.metrics import average_precision_score, roc_auc_score, fbeta_score, precision_recall_curve
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False
BETA = 2.0
HIDDEN_DIM = 128
DROPOUT = 0.3

def _check():
    if not _AVAILABLE:
        raise ImportError('torch и torch_geometric обязательны для CARE-GNN')

class FraudCAREGNN(nn.Module if _AVAILABLE else object):

    def __init__(self, in_channels: int, hidden: int=HIDDEN_DIM, n_relations: int=1, dropout: float=DROPOUT, use_sim_gate: bool=False) -> None:
        _check()
        super().__init__()
        self.hidden = hidden
        self.n_relations = n_relations
        self.use_sim_gate = use_sim_gate
        self.dropout = dropout
        out_dim = hidden // 2
        self.convs1 = nn.ModuleList([SAGEConv(in_channels, hidden) for _ in range(n_relations)])
        self.bns1 = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(n_relations)])
        self.rel_att = nn.Linear(hidden, 1, bias=False)
        self.conv2 = SAGEConv(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.conv3 = SAGEConv(hidden, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.skip = nn.Linear(in_channels, out_dim, bias=False)
        if use_sim_gate:
            self.sim_thresholds = nn.Parameter(torch.zeros(n_relations))
        self.head = nn.Linear(out_dim, 2)

    def _filter_edges(self, x: 'torch.Tensor', edge_index: 'torch.Tensor', rel_idx: int) -> 'torch.Tensor':
        if not self.use_sim_gate or edge_index.size(1) == 0:
            return edge_index
        src, dst = edge_index
        x_norm = F.normalize(x.detach(), p=2, dim=-1)
        sim = (x_norm[src] * x_norm[dst]).sum(dim=-1)
        threshold = float(torch.sigmoid(self.sim_thresholds[rel_idx]).item())
        return edge_index[:, sim >= threshold]

    def forward(self, x: 'torch.Tensor', edge_indices_list: 'list[torch.Tensor]', edge_index_all: 'torch.Tensor') -> 'torch.Tensor':
        residual = self.skip(x)
        att_logits_list = []
        with torch.no_grad():
            for i, (conv, bn) in enumerate(zip(self.convs1, self.bns1)):
                ei = self._filter_edges(x, edge_indices_list[i], i)
                h_i = F.relu(bn(conv(x, ei))) if ei.size(1) > 0 else torch.zeros(x.size(0), self.hidden, device=x.device)
                att_logits_list.append(self.rel_att(h_i))
                del h_i
        att_logits = torch.cat(att_logits_list, dim=1)
        att_weights = torch.softmax(att_logits, dim=1)
        del att_logits_list, att_logits
        h = torch.zeros(x.size(0), self.hidden, device=x.device)
        for i, (conv, bn) in enumerate(zip(self.convs1, self.bns1)):
            ei = self._filter_edges(x, edge_indices_list[i], i)
            h_i = F.relu(bn(conv(x, ei))) if ei.size(1) > 0 else torch.zeros(x.size(0), self.hidden, device=x.device)
            h = h + att_weights[:, i:i + 1].detach() * h_i
            del h_i
        del att_weights
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.bn2(self.conv2(h, edge_index_all)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.bn3(self.conv3(h, edge_index_all)))
        return self.head(h + residual)

    def predict_proba(self, x: 'torch.Tensor', edge_indices_list: 'list[torch.Tensor]', edge_index_all: 'torch.Tensor') -> 'torch.Tensor':
        return torch.softmax(self.forward(x, edge_indices_list, edge_index_all), dim=-1)[:, 1]

def _fbeta_sweep(y_true: np.ndarray, y_prob: np.ndarray, beta: float=BETA):
    best_th, best_fb = (0.5, -1.0)
    for th in np.arange(0.01, 0.99, 0.01):
        fb = fbeta_score(y_true, (y_prob >= th).astype(int), beta=beta, zero_division=0)
        if fb > best_fb:
            best_fb, best_th = (fb, float(th))
    return (best_th, best_fb)

def train_care_gnn(data: 'object', relation_types: 'list[str]', in_channels: int, hidden: int=HIDDEN_DIM, dropout: float=DROPOUT, lr: float=0.001, epochs: int=80, early_stopping_patience: int=20, output_dir: 'str | Path'='outputs/gnn/care', feature_cols: 'list[str] | None'=None, use_sim_gate: bool=False, skip_if_checkpoint: bool=True) -> 'tuple[FraudCAREGNN, dict]':
    _check()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / 'care_meta.json'
    weights_path = out / 'best_model.pt'
    hist_path = out / 'care_history.json'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_relations = len(relation_types)
    if skip_if_checkpoint and meta_path.exists() and weights_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('in_channels') == in_channels and meta.get('hidden') == hidden and (abs(float(meta.get('dropout', 0)) - dropout) < 1e-06) and (meta.get('n_relations') == n_relations):
            model = FraudCAREGNN(in_channels, hidden, n_relations, dropout, use_sim_gate).to(device)
            state = torch.load(weights_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.eval()
            history = json.loads(hist_path.read_text()) if hist_path.exists() else {'skipped': True}
            history['skipped'] = True
            print(f"  CARE-GNN: загружен из чекпойнта  best_val_Fβ={meta.get('best_val_fbeta', 0):.4f}")
            return (model, history)
    edge_indices_by_type: dict = getattr(data, 'edge_indices_by_type', {})
    edge_indices_list = []
    for rt in relation_types:
        ei = edge_indices_by_type.get(rt, torch.zeros((2, 0), dtype=torch.long))
        edge_indices_list.append(ei.to(device))
    edge_index_all = data.edge_index.to(device)
    x = data.x.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    y_train = y[train_mask].cpu().numpy()
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    w = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float, device=device)
    model = FraudCAREGNN(in_channels, hidden, n_relations, dropout, use_sim_gate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_fb, best_epoch, patience_cnt = (-1.0, 0, 0)
    history: dict = {'train_loss': [], 'val_fbeta': [], 'val_prauc': []}
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out_logits = model(x, edge_indices_list, edge_index_all)
        loss = F.cross_entropy(out_logits[train_mask], y[train_mask], weight=w)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            probs_all_t = torch.softmax(model(x, edge_indices_list, edge_index_all), dim=-1)[:, 1]
        val_y = y[val_mask].cpu().numpy()
        val_prob = probs_all_t[val_mask].cpu().numpy()
        val_prauc = average_precision_score(val_y, val_prob)
        _, val_fb = _fbeta_sweep(val_y, val_prob)
        history['train_loss'].append(float(loss.item()))
        history['val_fbeta'].append(float(val_fb))
        history['val_prauc'].append(float(val_prauc))
        if True:
            print(f'Epoch {epoch:3d} | loss={loss.item():.4f} | val F_β={val_fb:.4f} | val PR-AUC={val_prauc:.4f}')
        if val_fb > best_val_fb:
            best_val_fb, best_epoch, patience_cnt = (val_fb, epoch, 0)
            torch.save(model.state_dict(), weights_path)
        else:
            patience_cnt += 1
            if patience_cnt >= early_stopping_patience:
                print(f'  Early stopping на эпохе {epoch} (best={best_epoch}, Fβ={best_val_fb:.4f})')
                break
    print(f'\nBest epoch: {best_epoch}  F_β={best_val_fb:.4f}')
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    meta = {'model_type': 'care_gnn', 'in_channels': in_channels, 'hidden': hidden, 'dropout': dropout, 'n_relations': n_relations, 'relation_types': list(relation_types), 'use_sim_gate': use_sim_gate, 'feature_cols': list(feature_cols or []), 'best_epoch': best_epoch, 'best_val_fbeta': best_val_fb}
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    hist_path.write_text(json.dumps(history, ensure_ascii=False), encoding='utf-8')
    return (model, history)

def care_gnn_predict(model: 'FraudCAREGNN', data: 'object', relation_types: 'list[str]') -> np.ndarray:
    _check()
    device = next(model.parameters()).device
    edge_indices_by_type: dict = getattr(data, 'edge_indices_by_type', {})
    edge_indices_list = [edge_indices_by_type.get(rt, torch.zeros((2, 0), dtype=torch.long)).to(device) for rt in relation_types]
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(data.x.to(device), edge_indices_list, data.edge_index.to(device))
    return probs.cpu().numpy()
