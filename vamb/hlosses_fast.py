import vamb.hier as hier
from functools import partial
from multiprocessing import reduction
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _apply_to_maybe(fn: Callable, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return fn(x) if x is not None else None


class Sum(nn.Module):
    """Implements sum_xxx as an object. Avoids re-computation."""

    def __init__(
            self,
            tree: hier.Hierarchy,
            transpose: bool,
            subset: Optional[np.ndarray] = None,
            # leaf_only: bool = False,
            exclude_root: bool = False,
            strict: bool = False):
        super().__init__()
        # The value matrix[i, j] is true if i is an ancestor of j.
        # Take transpose for sum over descendants.
        matrix = tree.ancestor_mask(strict=strict)
        if subset is not None:
            matrix = matrix[:, subset]
        if exclude_root:
            matrix = matrix[1:, :]
        if transpose:
            matrix = matrix.T
        matrix = torch.from_numpy(matrix).type(torch.get_default_dtype())
        self.matrix = matrix

    def _apply(self, fn):
        super()._apply(fn)
        self.matrix = fn(self.matrix)
        return self

    def forward(self, values: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # TODO: Re-order dimensions to make this work with dim != -1.
        assert dim in (-1, values.ndim - 1)
        return torch.tensordot(values, self.matrix, dims=1)

SumAncestors = partial(Sum, transpose=False)
SumDescendants = partial(Sum, transpose=True)

class HierSoftmaxCrossEntropy(nn.Module):
    """Implements cross-entropy for YOLO-style conditional softmax. Avoids re-computation.

    Supports integer label targets or distribution targets.
    """

    def __init__(
            self,
            tree: hier.Hierarchy,
            with_leaf_targets: bool = False,
            label_smoothing: float = 0.0,
            node_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.label_smoothing = label_smoothing
        if with_leaf_targets:
            self.label_order = torch.from_numpy(tree.leaf_subset())
            self.num_labels = len(self.label_order)
        else:
            self.label_order = None
            self.num_labels = tree.num_nodes()
        self.hier_cond_log_softmax = HierCondLogSoftmax(tree)
        self.sum_label_descendants = SumDescendants(tree, subset=self.label_order)
        self.prior = torch.from_numpy(hier.uniform_leaf(tree))
        self.node_weight = node_weight

    def _apply(self, fn):
        super()._apply(fn)
        self.label_order = _apply_to_maybe(fn, self.label_order)
        self.hier_cond_log_softmax = self.hier_cond_log_softmax._apply(fn)
        self.sum_label_descendants = self.sum_label_descendants._apply(fn)
        self.prior = fn(self.prior)
        self.node_weight = _apply_to_maybe(fn, self.node_weight)
        return self

    def forward(self, scores: torch.Tensor, labels: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert labels.ndim in [scores.ndim, scores.ndim - 1]
        assert dim in (-1, scores.ndim - 1)
        # Convert labels to one-hot if they are not.
        if labels.ndim < scores.ndim:
            labels = F.one_hot(labels, self.num_labels)
        labels = labels.type(torch.get_default_dtype())
        q = self.sum_label_descendants(labels)
        if self.label_smoothing:
            q = (1 - self.label_smoothing) * q + self.label_smoothing * self.prior
        log_cond_p = self.hier_cond_log_softmax(scores, dim=-1)
        xent = q * -log_cond_p
        if self.node_weight is not None:
            xent = xent * self.node_weight
        xent = torch.sum(xent, dim=-1)
        return torch.mean(xent)


def hier_cond_log_softmax(
        tree: hier.Hierarchy,
        scores: torch.Tensor,
        dim: int = -1) -> torch.Tensor:
    """Returns log-likelihood of each node given its parent."""
    # Split scores into softmax for each internal node over its children.
    # Convert from [s[0], s[1], ..., s[n-1]]
    # to [[s[0], ..., s[k-1], -inf, -inf, ...],
    #     ...
    #     [..., s[n-1], -inf, -inf, ...]].
    # Use index_copy with flat_index, then reshape and compute log_softmax.
    # Then re-flatten and use index_select with flat_index.
    # This is faster than using torch.split() and map(log_softmax, ...).
    assert dim == -1 or dim == scores.ndim - 1
    num_nodes = tree.num_nodes()
    num_internal = tree.num_internal_nodes()
    node_to_children = tree.children()
    cond_children = [node_to_children[x] for x in tree.internal_subset()]
    cond_num_children = list(map(len, cond_children))
    max_num_children = max(cond_num_children)
    # TODO: Use _split_and_pad?
    row_index = np.concatenate([np.full(n, i) for i, n in enumerate(cond_num_children)])
    col_index = np.concatenate([np.arange(n) for n in cond_num_children])
    flat_index = row_index * max_num_children + col_index
    child_index = np.concatenate(cond_children)

    device = scores.device
    flat_index = torch.from_numpy(flat_index).to(device)
    child_index = torch.from_numpy(child_index).to(device)
    input_shape = list(scores.shape)
    flat_shape = [*input_shape[:-1], num_internal * max_num_children]
    # Pad with -inf for log_softmax.
    # flat[..., flat_index] = scores
    flat = torch.full(flat_shape, -torch.inf, device=device).index_copy(
        -1, flat_index, scores)
    split_shape = [*input_shape[:-1], num_internal, max_num_children]
    child_scores = flat.reshape(split_shape)
    child_log_p = F.log_softmax(child_scores, dim=-1)
    child_log_p = child_log_p.reshape(flat_shape)
    output_shape = [*input_shape[:-1], num_nodes]
    # log_cond_p[..., child_index] = child_log_p[..., flat_index]
    log_cond_p = torch.zeros(output_shape, device=device).index_copy(
        -1, child_index, child_log_p.index_select(-1, flat_index))
    return log_cond_p


def hier_log_softmax(
        tree: hier.Hierarchy,
        scores: torch.Tensor,
        dim: int = -1) -> torch.Tensor:
    """Returns log-likelihood for conditional softmax."""
    # Finally, take sum over ancestor conditionals to obtain likelihoods.
    assert dim in (-1, scores.ndim - 1)
    log_cond_p = hier_cond_log_softmax(tree, scores, dim=dim)
    # TODO: Use functional form here?
    device = scores.device
    sum_ancestors_fn = SumAncestors(tree, exclude_root=True).to(device)
    return sum_ancestors_fn(log_cond_p, dim=-1)


class HierCondLogSoftmax(nn.Module):
    """Implements hier_cond_log_softmax as an object. Avoids re-computation."""

    def __init__(self, tree: hier.Hierarchy):
        super().__init__()
        num_nodes = tree.num_nodes()
        num_internal = tree.num_internal_nodes()
        node_to_children = tree.children()
        cond_children = [node_to_children[x] for x in tree.internal_subset()]
        cond_num_children = list(map(len, cond_children))
        max_num_children = max(cond_num_children)
        # TODO: Use _split_and_pad?
        row_index = np.concatenate([np.full(n, i) for i, n in enumerate(cond_num_children)])
        col_index = np.concatenate([np.arange(n) for n in cond_num_children])
        flat_index = torch.from_numpy(row_index * max_num_children + col_index)
        child_index = torch.from_numpy(np.concatenate(cond_children))

        self.num_nodes = num_nodes
        self.num_internal = num_internal
        self.max_num_children = max_num_children
        self.flat_index = flat_index
        self.child_index = child_index

    def _apply(self, fn):
        super()._apply(fn)
        self.flat_index = fn(self.flat_index)
        self.child_index = fn(self.child_index)
        return self

    def forward(self, scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim in (-1, scores.ndim - 1)
        device = scores.device
        input_shape = list(scores.shape)
        flat_shape = [*input_shape[:-1], self.num_internal * self.max_num_children]
        # Pad with -inf for log_softmax.
        # flat[..., flat_index] = scores
        flat = torch.full(flat_shape, -torch.inf, device=device).index_copy(
            -1, self.flat_index, scores)
        split_shape = [*input_shape[:-1], self.num_internal, self.max_num_children]
        child_scores = flat.reshape(split_shape)
        child_log_p = F.log_softmax(child_scores, dim=-1)
        child_log_p = child_log_p.reshape(flat_shape)
        output_shape = [*input_shape[:-1], self.num_nodes]
        # log_cond_p[..., child_index] = child_log_p[..., flat_index]
        log_cond_p = torch.zeros(output_shape, device=device).index_copy(
            -1, self.child_index, child_log_p.index_select(-1, self.flat_index))
        return log_cond_p


class HierLogSoftmax(nn.Module):
    """Implements hier_log_softmax as an object. Avoids re-computation."""

    def __init__(self, tree: hier.Hierarchy):
        super().__init__()
        self.cond_log_softmax = HierCondLogSoftmax(tree)
        self.sum_ancestors_fn = SumAncestors(tree, exclude_root=False)

    def _apply(self, fn):
        super()._apply(fn)
        self.cond_log_softmax = self.cond_log_softmax._apply(fn)
        self.sum_ancestors_fn = self.sum_ancestors_fn._apply(fn)
        return self

    def forward(self, scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
        log_cond_p = self.cond_log_softmax(scores, dim=dim)
        return self.sum_ancestors_fn(log_cond_p, dim=dim)
