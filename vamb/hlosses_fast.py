import vamb.hier as hier
from functools import partial
from typing import Callable, Optional, Sequence
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
        strict: bool = False,
    ):
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
        node_weight: Optional[torch.Tensor] = None,
    ):
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

    def forward(
        self, scores: torch.Tensor, labels: torch.Tensor, dim: int = -1
    ) -> torch.Tensor:
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
    tree: hier.Hierarchy, scores: torch.Tensor, dim: int = -1
) -> torch.Tensor:
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
        -1, flat_index, scores
    )
    split_shape = [*input_shape[:-1], num_internal, max_num_children]
    child_scores = flat.reshape(split_shape)
    child_log_p = F.log_softmax(child_scores, dim=-1)
    child_log_p = child_log_p.reshape(flat_shape)
    output_shape = [*input_shape[:-1], num_nodes]
    # log_cond_p[..., child_index] = child_log_p[..., flat_index]
    log_cond_p = torch.zeros(output_shape, device=device).index_copy(
        -1, child_index, child_log_p.index_select(-1, flat_index)
    )
    return log_cond_p


def hier_log_softmax(
    tree: hier.Hierarchy, scores: torch.Tensor, dim: int = -1
) -> torch.Tensor:
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
        row_index = np.concatenate(
            [np.full(n, i) for i, n in enumerate(cond_num_children)]
        )
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
            -1, self.flat_index, scores
        )
        split_shape = [*input_shape[:-1], self.num_internal, self.max_num_children]
        child_scores = flat.reshape(split_shape)
        child_log_p = F.log_softmax(child_scores, dim=-1)
        child_log_p = child_log_p.reshape(flat_shape)
        output_shape = [*input_shape[:-1], self.num_nodes]
        # log_cond_p[..., child_index] = child_log_p[..., flat_index]
        log_cond_p = torch.zeros(output_shape, device=device).index_copy(
            -1, self.child_index, child_log_p.index_select(-1, self.flat_index)
        )
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


def multilabel_log_likelihood(
        scores: torch.Tensor,
        dim: int = -1,
        insert_root: bool = False,
        replace_root: bool = False,
        temperature: Optional[float] = None) -> torch.Tensor:
    assert not (insert_root and replace_root)
    assert dim in (-1, scores.ndim - 1)
    device = scores.device
    if temperature:
        scores = scores / temperature
    logp = F.logsigmoid(scores)
    if insert_root:
        zero = torch.zeros((*scores.shape[:-1], 1), device=device)
        logp = torch.cat([zero, logp], dim=-1)
    elif replace_root:
        zero = torch.zeros((*scores.shape[:-1], 1), device=device)
        logp = logp.index_copy(-1, torch.tensor([0], device=device), zero)
    return logp


class RandomCut(nn.Module):
    """Samples a random cut of the tree.

    Starts from the root and samples from Bernoulli for each node.
    If a node's sample is 0, the tree is terminated at that node.

    Returns a binary mask for the leaf nodes of the cut.
    """

    def __init__(self, tree: hier.Hierarchy, cut_prob: float, permit_root_cut: bool = False):
        super().__init__()
        self.num_nodes = tree.num_nodes()
        self.cut_prob = cut_prob
        self.permit_root_cut = permit_root_cut
        self.sum_ancestors = SumAncestors(tree)
        self.node_parent = torch.from_numpy(tree.parents(root_loop=True))

    def _apply(self, fn):
        super()._apply(fn)
        self.sum_ancestors._apply(fn)
        self.node_parent = fn(self.node_parent)
        return self

    def forward(self, batch_shape: Sequence[int]) -> torch.Tensor:
        device = self.node_parent.device
        # Random Bernoulli for every node.
        drop = torch.bernoulli(torch.full((*batch_shape, self.num_nodes), self.cut_prob))
        drop = drop.to(device=device)
        if not self.permit_root_cut:
            drop[..., 0] = 0
        # Check whether to keep all ancestors of each node (drop zero ancestors).
        subtree_mask = (self.sum_ancestors(drop, dim=-1) == 0)
        # Dilate to keep nodes whose parents belong to subtree.
        subtree_mask = subtree_mask[..., self.node_parent]
        subtree_mask[..., 0] = True  # Root is always kept.
        # Count number of children (do not count parent as child of itself).
        num_children = torch.zeros((*batch_shape, self.num_nodes), device=device)
        num_children = num_children.index_add(-1, self.node_parent[1:], subtree_mask[..., 1:].float())
        # Find boundary of subtree.
        boundary = subtree_mask & (num_children == 0)
        return boundary


class RandomCutLoss(nn.Module):
    """Cross-entropy loss using the leaf nodes of a random cut.
    
    As described in "Deep RTC" (Wu et al., 2020).
    """

    def __init__(
            self,
            tree: hier.Hierarchy,
            cut_prob: float,
            permit_root_cut: bool = False,
            with_leaf_targets: bool = True):
        super().__init__()
        is_ancestor = tree.ancestor_mask(strict=False)
        print('is_ancestor', is_ancestor.shape)
        # label_to_targets[gt, pr] = 1 iff pr `is_ancestor` gt
        label_to_targets = is_ancestor.T
        if with_leaf_targets:
            label_order = tree.leaf_subset()
            label_to_targets = label_to_targets[label_order]
        else:
            # Need to use FlatSoftmaxNLL?
            raise NotImplementedError

        self.random_cut_fn = RandomCut(tree, cut_prob=cut_prob, permit_root_cut=permit_root_cut)
        self.label_to_targets = torch.from_numpy(label_to_targets)

    def _apply(self, fn):
        super()._apply(fn)
        self.random_cut_fn._apply(fn)
        self.label_to_targets = fn(self.label_to_targets)
        return self

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = torch.argmax(labels, dim=1)
        cut = self.random_cut_fn(scores.shape[:-1])
        targets = self.label_to_targets[labels.long(), :]
        # No loss for root node.
        cut = cut[..., 1:]
        targets = targets[..., 1:]
        # Obtain targets in cut subset.
        cut_targets = (cut & targets)
        assert torch.all(torch.sum(cut_targets, dim=-1) == 1)
        # loss = F.cross_entropy(cut_scores, cut_targets, reduction='none')
        neg_inf = torch.tensor(-torch.inf, device=scores.device)
        zero = torch.tensor(0.0, device=scores.device)
        pos_score = torch.sum(torch.where(cut_targets, scores, zero), dim=-1)
        loss = -pos_score + torch.logsumexp(torch.where(cut, scores, neg_inf), dim=-1)
        return torch.mean(loss)


class LCAMetric:

    def __init__(self, tree: hier.Hierarchy, value: np.ndarray):
        self.value = value
        self.find_lca = hier.FindLCA(tree)

    def value_at_lca(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        return self.value[lca]

    def value_at_gt(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        # TODO: Avoid broadcasting of unused array?
        gt, _ = np.broadcast_arrays(gt, pr)
        return self.value[gt]

    def value_at_pr(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        # TODO: Avoid broadcasting of unused array?
        _, pr = np.broadcast_arrays(gt, pr)
        return self.value[pr]

    def deficient(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        return self.value[gt] - self.value[lca]

    def excess(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        return self.value[pr] - self.value[lca]

    def dist(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        excess = self.value[pr] - self.value[lca]
        deficient = self.value[gt] - self.value[lca]
        return excess + deficient

    def recall(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        gt_value = self.value[gt]
        lca_value = self.value[lca]
        with np.errstate(invalid='ignore'):
            return np.where((lca_value == 0) & (gt_value == 0), 1.0, lca_value / gt_value)

    def precision(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        pr_value = self.value[pr]
        lca_value = self.value[lca]
        with np.errstate(invalid='ignore'):
            return np.where((lca_value == 0) & (pr_value == 0), 1.0, lca_value / pr_value)

    def f1(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        gt_value = self.value[gt]
        pr_value = self.value[pr]
        lca_value = self.value[lca]
        with np.errstate(invalid='ignore'):
            r = np.where((lca_value == 0) & (gt_value == 0), 1.0, lca_value / gt_value)
            p = np.where((lca_value == 0) & (pr_value == 0), 1.0, lca_value / pr_value)
        with np.errstate(divide='ignore'):
            return 2 / (1/r + 1/p)


class MarginLoss(nn.Module):
    """Computes soft or hard margin loss for given a margin function."""

    def __init__(
            self, tree: hier.Hierarchy,
            with_leaf_targets: bool,
            hardness: str = 'soft',
            margin: str = 'depth_dist',
            tau: float = 1.0):
        super().__init__()
        if hardness not in ('soft', 'hard'):
            raise ValueError('unknown hardness', hardness)
        n = tree.num_nodes()
        label_order = tree.leaf_subset() if with_leaf_targets else np.arange(n)

        # Construct array label_margin[gt_label, pr_node].
        if margin in ('edge_dist', 'depth_dist'):
            # label_margin = metrics.edge_dist(tree, label_order[:, None], np.arange(n)[None, :])
            depth = tree.depths()
            margin_arr = LCAMetric(tree, depth).dist(label_order[:, None], np.arange(n))
        elif margin == 'incorrect':
            is_ancestor = tree.ancestor_mask()
            is_correct = is_ancestor[:, label_order].T
            margin_arr = 1 - is_correct
        elif margin == 'info_dist':
            # TODO: Does natural log make most sense here?
            info = np.log(tree.num_leaf_nodes() / tree.num_leaf_descendants())
            margin_arr = LCAMetric(tree, info).dist(label_order[:, None], np.arange(n))
        elif margin == 'depth_deficient':
            depth = tree.depths()
            margin_arr = LCAMetric(tree, depth).deficient(label_order[:, None], np.arange(n))
        elif margin == 'log_depth_f1_error':
            depth = tree.depths()
            margin_arr = np.log(1 - LCAMetric(tree, depth).f1(label_order[:, None], np.arange(n)))
        else:
            raise ValueError('unknown margin', margin)

        # correct_margins = margin_arr[np.arange(len(label_order)), label_order]
        # if not np.all(correct_margins == 0):
        #     raise ValueError('margin with self is not zero', correct_margins)

        self.hardness = hardness
        self.tau = tau
        self.label_order = torch.from_numpy(label_order)
        self.margin = torch.from_numpy(margin_arr)

    def _apply(self, fn):
        super()._apply(fn)
        self.label_order = fn(self.label_order)
        self.margin = fn(self.margin)
        return self

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = torch.argmax(labels, dim=1)
        label_node = labels if self.label_order is None else self.label_order[labels.long()]
        label_score = scores.gather(-1, label_node.unsqueeze(-1)).squeeze(-1)
        label_margin = self.margin[labels.long(), :]
        if self.hardness == 'soft':
            loss = -label_score + torch.logsumexp(scores + self.tau * label_margin, axis=-1)
        elif self.hardness == 'hard':
            # loss = -label_score + torch.max(torch.relu(scores + self.tau * label_margin), axis=-1)[0]
            loss = torch.relu(torch.max(scores - label_score.unsqueeze(-1) + self.tau * label_margin, axis=-1)[0])
        else:
            assert False
        return torch.mean(loss)


class FlatSoftmaxNLL(nn.Module):
    """Like cross_entropy() but supports internal labels."""

    def __init__(self, tree, with_leaf_targets: bool = False, reduction: str = 'mean'):
        super().__init__()
        assert reduction in ('mean', 'none', None)
        if with_leaf_targets:
            raise ValueError('use F.cross_entropy() instead!')
        # The value is_ancestor[i, j] is true if node i is an ancestor of node j.
        is_ancestor = tree.ancestor_mask(strict=False)
        leaf_masks = is_ancestor[:, tree.leaf_mask()]
        self.leaf_masks = torch.from_numpy(leaf_masks)
        self.reduction = reduction

    def _apply(self, fn):
        super()._apply(fn)
        self.leaf_masks = fn(self.leaf_masks)
        return self

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = torch.argmax(labels, dim=1)
        logp_leaf = F.log_softmax(scores, dim=-1)
        # Obtain logp for leaf descendants, -inf for other nodes.
        label_leaf_mask = self.leaf_masks[labels.long(), :]
        inf = torch.tensor(torch.inf, device=scores.device)
        logp_descendants = torch.where(label_leaf_mask, logp_leaf, -inf)
        logp_label = torch.logsumexp(logp_descendants, dim=-1)
        loss = -logp_label
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss


def SumLeafDescendants(
        tree: hier.Hierarchy,
        **kwargs):
    return SumDescendants(tree, subset=tree.leaf_mask(), **kwargs)
