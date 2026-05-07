from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import numpy as np
try:
    import torch
    from torch_geometric.explain import Explainer, GNNExplainer
    from torch_geometric.utils import k_hop_subgraph, subgraph as tg_subgraph
    _EXPLAINER_AVAILABLE = True
except ImportError:
    try:
        import torch
        from torch_geometric.nn import GNNExplainer
        from torch_geometric.utils import k_hop_subgraph, subgraph as tg_subgraph
        _EXPLAINER_AVAILABLE = True
        _EXPLAINER_LEGACY = True
    except ImportError:
        _EXPLAINER_AVAILABLE = False
        _EXPLAINER_LEGACY = False
else:
    _EXPLAINER_LEGACY = False

def clear_message_passing_explain(model: 'torch.nn.Module') -> None:
    try:
        from torch_geometric.nn.conv.message_passing import MessagePassing
    except ImportError:
        return
    for m in model.modules():
        if isinstance(m, MessagePassing):
            m.explain = False
            m._edge_mask = None
            m._loop_mask = None

class GNNExplainerWrapper:

    def __init__(self, model, data, n_epochs: int=50, **kwargs) -> None:
        if not _EXPLAINER_AVAILABLE:
            raise ImportError('torch_geometric.explain (or torch_geometric.nn.GNNExplainer) is required. Install: pip install torch_geometric')
        explain_num_hops = int(kwargs.pop('explain_num_hops', 2))
        explain_max_nodes = int(kwargs.pop('explain_max_nodes', 2500))
        full_graph_explain_max_nodes = int(kwargs.pop('full_graph_explain_max_nodes', 8000))
        full_graph_explain_max_edges = int(kwargs.pop('full_graph_explain_max_edges', 250000))
        if kwargs:
            raise TypeError(f'GNNExplainerWrapper: неизвестные аргументы {sorted(kwargs)!r}')
        self.model = model
        self.data = data
        self.n_epochs = n_epochs
        self.explain_num_hops = explain_num_hops
        self.explain_max_nodes = explain_max_nodes
        self._full_graph_explain_max_nodes = full_graph_explain_max_nodes
        self._full_graph_explain_max_edges = full_graph_explain_max_edges
        self._explain_cache: dict[int, tuple[object, int]] = {}
        self._explainer = self._build_explainer()

    def _build_explainer(self):
        if not _EXPLAINER_LEGACY:
            return Explainer(model=self.model, algorithm=GNNExplainer(epochs=self.n_epochs), explanation_type='model', node_mask_type='attributes', edge_mask_type=None, model_config=dict(mode='multiclass_classification', task_level='node', return_type='probs'))
        else:
            return GNNExplainer(self.model, epochs=self.n_epochs, return_type='prob')

    def _explanation_subgraph(self, node_idx: int) -> tuple['torch.Tensor', 'torch.Tensor', int]:
        device = self.data.x.device
        ei = self.data.edge_index
        n_nodes = int(self.data.num_nodes)
        x = self.data.x
        cap = self.explain_max_nodes
        subset, ei_sub, mapping, _ = k_hop_subgraph(node_idx, self.explain_num_hops, ei, num_nodes=n_nodes, relabel_nodes=True)
        if subset.numel() > cap:
            subset, ei_sub, mapping, _ = k_hop_subgraph(node_idx, 1, ei, num_nodes=n_nodes, relabel_nodes=True)
        if subset.numel() > cap:
            row, col = ei
            mask = (row == node_idx) | (col == node_idx)
            neigh = torch.unique(torch.cat([row[mask], col[mask]]))
            neigh = neigh[neigh != node_idx]
            if neigh.numel() > cap - 1:
                perm = torch.randperm(neigh.numel(), device=device)[:cap - 1]
                neigh = neigh[perm]
            subset = torch.cat([torch.tensor([node_idx], device=device, dtype=torch.long), neigh])
            subset = torch.sort(subset).values
            ei_sub, _ = tg_subgraph(subset, ei, relabel_nodes=True, num_nodes=n_nodes)
            mapping = (subset == node_idx).nonzero(as_tuple=True)[0][0]
        x_sub = x[subset]

        def _idx(m: object) -> int:
            return int(m.item()) if hasattr(m, 'item') else int(m)
        return (x_sub, ei_sub, _idx(mapping))

    def _use_full_graph_for_explain(self) -> bool:
        n = int(self.data.num_nodes)
        e = int(self.data.edge_index.shape[1])
        return n <= self._full_graph_explain_max_nodes and e <= self._full_graph_explain_max_edges

    def _gradient_explanation(self, x: 'torch.Tensor', edge_index: 'torch.Tensor', local_idx: int):
        import torch
        clear_message_passing_explain(self.model)
        self.model.train(False)
        x_in = x.detach().clone().float().requires_grad_(True)
        out = self.model(x_in, edge_index)
        prob_fraud = torch.softmax(out, dim=-1)[local_idx, 1]
        prob_fraud.backward()
        g = x_in.grad
        if g is None:
            raise RuntimeError('gradient explanation: grad is None')
        mask = g.detach().abs()
        return SimpleNamespace(node_mask=mask)

    def explain_node(self, node_idx: int) -> object:
        if node_idx in self._explain_cache:
            return self._explain_cache[node_idx][0]
        clear_message_passing_explain(self.model)
        try:
            full_graph = self._use_full_graph_for_explain()
            if full_graph:
                x, ei, local_idx = (self.data.x, self.data.edge_index, int(node_idx))
            else:
                x, ei, local_idx = self._explanation_subgraph(int(node_idx))
            if not _EXPLAINER_LEGACY:
                if not full_graph:
                    explanation = self._gradient_explanation(x, ei, local_idx)
                else:
                    try:
                        explanation = self._explainer(x, ei, index=local_idx)
                    except AssertionError:
                        explanation = self._gradient_explanation(x, ei, local_idx)
            else:
                try:
                    node_feat_mask, edge_mask = self._explainer.explain_node(local_idx, x, ei)
                    explanation = (node_feat_mask, edge_mask)
                except Exception:
                    gm = self._gradient_explanation(x, ei, local_idx).node_mask
                    explanation = (gm, None)
            self._explain_cache[node_idx] = (explanation, local_idx)
            return explanation
        finally:
            clear_message_passing_explain(self.model)

    def top_features(self, node_idx: int, feature_names: list[str], top_k: int=10) -> list[tuple[str, float]]:
        result = self.explain_node(node_idx)
        row_idx = self._explain_cache[node_idx][1]
        if not _EXPLAINER_LEGACY:
            if result.node_mask is None:
                return []
            raw = result.node_mask.cpu().numpy()
        else:
            raw = result[0].cpu().numpy()
        if raw.ndim == 1:
            mask = raw
        elif raw.shape[0] == 1:
            mask = raw[0]
        else:
            mask = raw[row_idx]
        importance = np.abs(mask).reshape(-1)
        n_feat = min(len(feature_names), int(importance.shape[0]))
        top_idx = np.argsort(importance[:n_feat])[::-1][:top_k]
        return [(feature_names[i], float(importance[i])) for i in top_idx]

    def visualize(self, node_idx: int, feature_names: list[str] | None=None, output_dir: str | Path='outputs/gnn', top_k: int=15) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib is required for visualization')
            return
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        result = self.explain_node(node_idx)
        row_idx = self._explain_cache[node_idx][1]
        if not _EXPLAINER_LEGACY:
            if result.node_mask is None:
                print('  node_mask отсутствует — визуализация пропущена.')
                return
            raw = result.node_mask.cpu().numpy()
        else:
            raw = result[0].cpu().numpy()
        if raw.ndim == 1:
            mask = raw
        elif raw.shape[0] == 1:
            mask = raw[0]
        else:
            mask = raw[row_idx]
        importance = np.abs(mask).reshape(-1)
        names = feature_names or [f'f{i}' for i in range(len(importance))]
        n_feat = len(names) if names else int(importance.shape[-1])
        n_feat = min(n_feat, int(importance.shape[-1]))
        top_idx = np.argsort(importance[:n_feat])[::-1][:top_k]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh([names[i] for i in reversed(top_idx)], [importance[i] for i in reversed(top_idx)], color='steelblue')
        ax.set_xlabel('|∂ P(fraud) / ∂ feature| (градиент на подграфе)')
        ax.set_title(f'Node {node_idx} — top {top_k} features')
        plt.tight_layout()
        save_path = Path(output_dir) / f'gnn_explain_node_{node_idx}.png'
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.show()
        plt.close('all')
        print(f'Saved: {save_path}')
