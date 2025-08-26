"""
The following code is a modification of a hierarchical losses implementation first published at https://github.com/jvlmdr/hiercls
"""

from functools import partial
import numpy as np
import collections
import csv
import itertools
from typing import Callable, Hashable, Optional, Sequence, Tuple, TextIO, cast

import networkx as nx
from numpy.typing import ArrayLike

import torch
from torch import nn
import torch.nn.functional as F


class Hierarchy:
    """Hierarchy of nodes 0, ..., n-1."""

    def __init__(self, parents):
        n = len(parents)
        assert np.all(parents[1:] < np.arange(1, n))
        self._parents = parents

    def num_nodes(self) -> int:
        return len(self._parents)

    def edges(self) -> list[tuple[int, int]]:
        return list(zip(self._parents[1:], itertools.count(1)))

    def parents(self, root_loop: bool = False) -> np.ndarray:
        if root_loop:
            return np.where(
                np.array(self._parents) >= 0,
                np.array(self._parents),
                np.arange(len(self._parents)),
            )
        else:
            return np.array(self._parents)

    def children(self) -> dict[int, np.ndarray]:
        result = collections.defaultdict(list)
        for i, j in self.edges():
            result[i].append(j)
        return {k: np.array(v, dtype=int) for k, v in result.items()}

    def num_children(self) -> np.ndarray:
        n = len(self._parents)
        unique, counts = np.unique(self._parents[1:], return_counts=True)
        result = np.zeros([n], dtype=int)
        result[unique] = counts
        return result

    def leaf_mask(self) -> np.ndarray:
        return self.num_children() == 0

    def leaf_subset(self) -> np.ndarray:
        (index,) = self.leaf_mask().nonzero()
        return index

    def internal_subset(self) -> np.ndarray:
        (index,) = np.logical_not(self.leaf_mask()).nonzero()
        return index

    def num_leaf_nodes(self) -> int:
        return np.count_nonzero(self.leaf_mask())

    def num_internal_nodes(self) -> int:
        return np.count_nonzero(np.logical_not(self.leaf_mask()))

    def num_conditionals(self) -> int:
        return np.count_nonzero(self.num_children() > 1)

    def depths(self) -> np.ndarray:
        return self.accumulate_ancestors(np.add, (self._parents >= 0).astype(int))

    def num_leaf_descendants(self) -> np.ndarray:
        return self.accumulate_descendants(np.add, self.leaf_mask().astype(int))

    def max_heights(self) -> np.ndarray:
        heights = np.zeros_like(self.depths())
        return self.accumulate_descendants(lambda u, v: max(u, v + 1), heights)

    def min_heights(self) -> np.ndarray:
        # Initialize leaf nodes to zero, internal nodes to upper bound.
        depths = self.depths()
        heights = np.where(self.leaf_mask(), 0, depths.max() - depths)
        return self.accumulate_descendants(lambda u, v: min(u, v + 1), heights)

    def accumulate_ancestors(self, func: Callable, values: ArrayLike) -> np.ndarray:
        # Start from root and move down.
        partials = np.array(values)
        for i, j in self.edges():
            partials[j] = func(partials[i], partials[j])
        return partials

    def accumulate_descendants(self, func: Callable, values: ArrayLike) -> np.ndarray:
        # Start from leaves and move up.
        partials = np.array(values)
        for i, j in reversed(self.edges()):
            partials[i] = func(partials[i], partials[j])
        return partials

    def ancestor_mask(self, strict=False) -> np.ndarray:
        n = len(self._parents)
        # If path exists i, ..., j then i < j and is_ancestor[i, j] is True.
        # Work with is_descendant instead to use consecutive blocks of memory.
        # Note that is_ancestor[i, j] == is_descendant[j, i].
        is_descendant = np.zeros([n, n], dtype=bool)
        if not strict:
            is_descendant[0, 0] = 1
        for i, j in self.edges():
            # Node i is parent of node j.
            assert i < j, "require edges in topological order"
            is_descendant[j, :] = is_descendant[i, :]
            if strict:
                is_descendant[j, i] = 1
            else:
                is_descendant[j, j] = 1
        is_ancestor = is_descendant.T
        return is_ancestor

    def paths(
        self,
        exclude_root: bool = False,
        exclude_self: bool = False,
    ) -> list[np.ndarray]:
        is_descendant = self.ancestor_mask(strict=exclude_self).T
        if exclude_root:
            paths = [np.flatnonzero(mask) + 1 for mask in is_descendant[:, 1:]]
        else:
            paths = [np.flatnonzero(mask) for mask in is_descendant]
        return paths

    def paths_padded(
        self, pad_value=-1, method: str = "constant", **kwargs
    ) -> np.ndarray:
        n = self.num_nodes()
        paths = self.paths(**kwargs)
        path_lens = list(map(len, paths))
        max_len = max(path_lens)
        if method == "constant":
            padded = np.full((n, max_len), pad_value, dtype=int)
        elif method == "self":
            padded = np.tile(np.arange(n)[:, None], max_len)
        else:
            raise ValueError("unknown pad method", method)
        row_index = np.concatenate([np.full(n, i) for i, n in enumerate(path_lens)])
        col_index = np.concatenate([np.arange(n) for n in path_lens])
        padded[row_index, col_index] = np.concatenate(paths)
        return padded

    def __str__(self, node_names: Optional[list[str]] = None) -> str:
        return format_tree(self, node_names)

    def to_networkx(self, keys: Optional[list] = None) -> nx.DiGraph:
        n = self.num_nodes()
        g = nx.DiGraph()
        if keys is None:
            keys = list(range(n))
        g.add_edges_from([(keys[i], keys[j]) for i, j in self.edges()])
        return g


def make_hierarchy_from_edges(
    pairs: Sequence[tuple[str, str]],
) -> tuple[Hierarchy, list[str]]:
    """Creates a hierarchy from a list of name pairs.

    The order of the edges determines the order of the nodes.
    (Each node except the root appears once as a child.)
    The root is placed first in the order.
    """
    num_edges = len(pairs)
    num_nodes = num_edges + 1
    # Data structures to populate from list of pairs.
    parents = np.full([num_nodes], -1, dtype=int)
    names: list[Optional[str]] = [None] * num_nodes
    name_to_index = {}
    # Set name of root from first pair.
    root, _ = pairs[0]
    names[0] = root
    name_to_index[root] = 0
    for r, (u, v) in enumerate(pairs):
        if v in name_to_index:
            raise ValueError("has multiple parents", v)
        i = name_to_index[u]
        j = r + 1
        parents[j] = i
        names[j] = v
        name_to_index[v] = j
    assert all(i is not None for i in names)
    return Hierarchy(parents), cast(list[str], names)


def load_edges(f: TextIO, delimiter=",") -> list[tuple[str, str]]:
    """Load from file containing (parent, node) pairs."""
    reader = csv.reader(f)
    pairs = []
    for row in reader:
        if not row:
            continue
        if len(row) != 2:
            raise ValueError("invalid row", row)
        pairs.append(tuple(row))
    return pairs


def rooted_subtree(tree: Hierarchy, nodes: np.ndarray) -> Hierarchy:
    """Finds the subtree that contains a subset of nodes."""
    # Check that root is present in subset.
    assert nodes[0] == 0
    # Construct a new list of parents.
    reindex = np.full([tree.num_nodes()], -1)
    reindex[nodes] = np.arange(len(nodes))
    parents = tree.parents()
    subtree_parents = np.where(parents[nodes] >= 0, reindex[parents[nodes]], -1)
    assert np.all(subtree_parents[1:] >= 0), "parent not in subset"
    # Ensure that parent appears before child.
    assert np.all(subtree_parents < np.arange(len(nodes)))
    return Hierarchy(subtree_parents)


def rooted_subtree_spanning(
    tree: Hierarchy, nodes: np.ndarray
) -> tuple[Hierarchy, np.ndarray]:
    nodes = ancestors_union(tree, nodes)
    subtree = rooted_subtree(tree, nodes)
    return subtree, nodes


def ancestors_union(tree: Hierarchy, node_subset: np.ndarray) -> np.ndarray:
    """Returns union of ancestors of nodes."""
    paths = tree.paths_padded(-1)
    paths = paths[node_subset]
    return np.unique(paths[paths >= 0])


def find_subset_index(base: list[Hashable], subset: list[Hashable]) -> np.ndarray:
    """Returns index of subset elements in base list (injective map)."""
    name_to_index = {x: i for i, x in enumerate(base)}
    return np.asarray([name_to_index[x] for x in subset], dtype=int)


def find_projection(tree: Hierarchy, node_subset: np.ndarray) -> np.ndarray:
    """Finds projection to nearest ancestor in subtree."""
    # Use paths rather than ancestor_mask to avoid large memory usage.
    assert np.all(node_subset >= 0)
    paths = tree.paths_padded(-1)
    # Find index in subset.
    reindex = np.full([tree.num_nodes()], -1)
    reindex[node_subset] = np.arange(len(node_subset))
    subset_paths = np.where(paths >= 0, reindex[paths], -1)
    deepest = _last_nonzero(subset_paths >= 0, axis=1)
    # Check that all ancestors are present.
    assert np.all(np.count_nonzero(subset_paths >= 0, axis=1) - 1 == deepest)
    return subset_paths[np.arange(tree.num_nodes()), deepest]


def _last_nonzero(x, axis):
    x = np.asarray(x, bool)
    assert np.all(np.any(x, axis=axis)), "at least one must be nonzero"
    # Find last element that is true.
    # (First element that is true in reversed array.)
    n = x.shape[axis]
    return (n - 1) - np.argmax(np.flip(x, axis), axis=axis)


def uniform_leaf(tree: Hierarchy) -> np.ndarray:
    """Returns a uniform distribution over leaf nodes."""
    is_ancestor = tree.ancestor_mask(strict=False)
    is_leaf = tree.leaf_mask()
    num_leaf_descendants = is_ancestor[:, is_leaf].sum(axis=1)
    return num_leaf_descendants / is_leaf.sum()


def uniform_cond(tree: Hierarchy) -> np.ndarray:
    """Returns a uniform distribution over child nodes at each conditional."""
    node_to_num_children = {k: len(v) for k, v in tree.children().items()}
    num_children = np.asarray(
        [node_to_num_children.get(x, 0) for x in range(tree.num_nodes())]
    )
    parent_index = tree.parents()
    # Root node has likelihood 1 and no parent.
    log_cond_p = np.concatenate([[0.0], -np.log(num_children[parent_index[1:]])])
    is_ancestor = tree.ancestor_mask(strict=False)
    log_p = np.dot(is_ancestor.T, log_cond_p)
    return np.exp(log_p)


def lca_depth(tree: Hierarchy, inds_a: np.ndarray, inds_b: np.ndarray) -> np.ndarray:
    """Returns the depth of the LCA node.

    Supports multi-dimensional index arrays.
    """
    paths = tree.paths_padded(exclude_root=True)
    paths_a = paths[inds_a]
    paths_b = paths[inds_b]
    return np.count_nonzero(
        ((paths_a == paths_b) & (paths_a >= 0) & (paths_b >= 0)), axis=-1
    )


def find_lca(tree: Hierarchy, inds_a: np.ndarray, inds_b: np.ndarray) -> np.ndarray:
    """Returns the index of the LCA node.

    Supports multi-dimensional index arrays.
    For example, to obtain an exhaustive table:
        n = tree.num_nodes()
        find_lca(tree, np.arange(n)[:, np.newaxis], np.arange(n)[np.newaxis, :])
    """
    paths = tree.paths_padded(exclude_root=False)
    paths_a = paths[inds_a]
    paths_b = paths[inds_b]
    num_common = np.count_nonzero(
        ((paths_a == paths_b) & (paths_a >= 0) & (paths_b >= 0)), axis=-1
    )
    return paths[inds_a, num_common - 1]


class FindLCA:
    def __init__(self, tree: Hierarchy):
        self.paths = tree.paths_padded(exclude_root=False)

    def __call__(self, inds_a: np.ndarray, inds_b: np.ndarray) -> np.ndarray:
        paths = self.paths
        paths_a = paths[inds_a]
        paths_b = paths[inds_b]
        num_common = np.count_nonzero(
            ((paths_a == paths_b) & (paths_a >= 0) & (paths_b >= 0)), axis=-1
        )
        return paths[inds_a, num_common - 1]


def truncate_at_lca(tree: Hierarchy, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
    """Truncates the prediction if a descendant of the ground-truth.

    Note that this calls find_lca().
    If calling repetitively, use `FindLCA` and `truncate_given_lca`.
    """
    lca = find_lca(tree, gt, pr)
    return truncate_given_lca(gt, pr, lca)


def truncate_given_lca(gt: np.ndarray, pr: np.ndarray, lca: np.ndarray) -> np.ndarray:
    """Truncates the prediction if a descendant of the ground-truth."""
    return np.where(gt == lca, gt, pr)


def format_tree(
    tree: Hierarchy, node_names: Optional[list[str]] = None, include_size: bool = False
) -> str:
    if node_names is None:
        node_names = [str(i) for i in range(tree.num_nodes())]

    node_to_children = tree.children()
    node_sizes = tree.num_leaf_descendants()

    def subtree(node, node_prefix, desc_prefix):
        name = node_names[node]
        size = node_sizes[node]
        text = f"{name} ({size})" if include_size and size > 1 else name
        yield node_prefix + text + "\n"
        children = node_to_children.get(node, ())
        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            yield from subtree(
                child,
                node_prefix=desc_prefix + ("└── " if is_last else "├── "),
                desc_prefix=desc_prefix + ("    " if is_last else "│   "),
            )

    return "".join(subtree(0, "", ""))


def level_nodes(tree: Hierarchy, extend: bool = False) -> list[np.ndarray]:
    node_depth = tree.depths()
    is_leaf = tree.leaf_mask()
    max_depth = np.max(node_depth)
    level_depth = np.arange(1, max_depth + 1)
    if not extend:
        level_masks = level_depth[:, None] == node_depth
    else:
        level_masks = (level_depth[:, None] == node_depth) | (
            (level_depth[:, None] > node_depth) & is_leaf
        )
    return [np.flatnonzero(mask) for mask in level_masks]


def level_successors(
    tree: Hierarchy,
) -> tuple[list[np.ndarray], list[list[np.ndarray]]]:
    """Returns the list of parents and their children at each depth."""
    node_depth = tree.depths()
    max_depth = np.max(node_depth)
    node_children = tree.children()

    # Get internal nodes at each level.
    level_parents = [[] for _ in range(max_depth)]
    level_children = [[] for _ in range(max_depth)]
    for u in tree.internal_subset():
        d = node_depth[u]
        level_parents[d].append(u)
        level_children[d].append(node_children[u])
    level_sizes = np.array(list(map(len, level_parents)))
    assert np.all(level_sizes > 0)

    level_parents = [np.array(x, dtype=int) for x in level_parents]
    return level_parents, level_children


def level_successors_padded(
    tree: Hierarchy,
    method: str = "constant",
    constant_value: int = -1,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Returns the list of parents and their children at each depth."""
    level_parents, level_children = level_successors(tree)
    level_children = [
        _pad_2d(x, dtype=np.dtype(int), method=method, constant_value=constant_value)
        for x in level_children
    ]
    return level_parents, level_children


def siblings(tree: Hierarchy) -> list[np.ndarray]:
    node_parent = tree.parents()
    node_children = tree.children()
    node_children[-1] = np.empty([0], dtype=int)
    return [_except(u, node_children[node_parent[u]]) for u in range(tree.num_nodes())]


def siblings_padded(
    tree: Hierarchy, method: str = "constant", constant_value: int = -1
) -> np.ndarray:
    ragged = siblings(tree)
    return _pad_2d(
        ragged, dtype=np.dtype(int), method=method, constant_value=constant_value
    )


def _except(v, x: np.ndarray) -> np.ndarray:
    return x[x != v]


def _pad_2d(
    x: list[np.ndarray], dtype: np.dtype, method: str = "constant", constant_value=None
) -> np.ndarray:
    num_rows = len(x)
    row_lens = list(map(len, x))
    num_cols = max(map(len, x))
    if method == "constant":
        assert constant_value is not None
        padded = np.full((num_rows, num_cols), constant_value, dtype=dtype)
    elif method == "first":
        padded = np.tile(
            np.asarray([row[0] for row in x], dtype=dtype)[:, None], num_cols
        )
    elif method == "last":
        padded = np.tile(
            np.asarray([row[-1] for row in x], dtype=dtype)[:, None], num_cols
        )
    elif method == "index":
        padded = np.tile(np.arange(num_rows, dtype=dtype)[:, None], num_cols)
    else:
        raise ValueError("unknown pad method", method)
    row_index = np.concatenate(
        [np.full(row_len, i) for i, row_len in enumerate(row_lens)]
    )
    col_index = np.concatenate([np.arange(row_len) for row_len in row_lens])
    padded[row_index, col_index] = np.concatenate(x)
    return padded


def most_confident_leaf(tree: Hierarchy, p: np.ndarray) -> np.ndarray:
    assert p.shape[-1] == tree.num_nodes()
    is_leaf = tree.leaf_mask()
    return argmax_where(p, is_leaf)


def max_info_majority_subtree(tree: Hierarchy, p: np.ndarray) -> np.ndarray:
    assert p.shape[-1] == tree.num_nodes()
    # -x is a monotonic mapping of -log(1/(x/n)) = log x - constant
    specificity = -tree.num_leaf_descendants()
    # Trivial nodes are allowed in the tree, but we will not predict them.
    # If the parent is equal to the child, it has the same score and specificity.
    # Maybe we should break ties in (specificity, p) using depth?
    not_trivial = tree.num_children() != 1
    return argmax_with_confidence(specificity, p, 0.5, not_trivial)


def pareto_optimal_predictions(
    info: np.ndarray,
    prob: np.ndarray,
    min_threshold: Optional[float] = None,
    condition: Optional[np.ndarray] = None,
    require_unique: bool = False,
) -> np.ndarray:
    """Finds the sequence of nodes that can be chosen by threshold.

    Returns nodes that are more specific than all more-confident predictions.
    This is equivalent to:
    (1) nodes such that there does not exist a node which is more confident and more specific,
    (2) nodes such that all nodes are less confident or less specific (*less than or equal).

    The resulting nodes are ordered descending by prob (and ascending by info).
    """
    assert prob.ndim == 1
    assert info.ndim == 1

    is_valid = np.ones(prob.shape, dtype=bool)
    if min_threshold is not None:
        is_valid = is_valid & (prob > min_threshold)
    if condition is not None:
        is_valid = is_valid & condition
    assert np.any(is_valid), "require at least one valid element"
    prob = prob[is_valid]
    info = info[is_valid]
    (valid_inds,) = np.nonzero(is_valid)

    # Order descending by prob then descending by info.
    # Note that np.lexsort() orders first by the last key.
    # (Performs stable sort from first key to last key.)
    order = np.lexsort((-info, -prob))
    prob = prob[order]
    info = info[order]

    max_info = np.maximum.accumulate(info)
    keep = (prob[1:] > prob[:-1]) | (info[1:] > max_info[:-1])
    keep = np.concatenate(([True], keep))

    if require_unique:
        if np.any(
            (prob[1:] == prob[:-1]) & (info[1:] == info[:-1]) & (keep[1:] | keep[:-1])
        ):
            raise ValueError("set is not unique")

    return valid_inds[order[keep]]


def argmax_where(
    value: np.ndarray, condition: np.ndarray, axis: int = -1, keepdims: bool = False
) -> np.ndarray:
    # Will raise an exception if not np.all(np.any(condition, axis=axis)).
    # return np.nanargmax(np.where(condition, value, np.nan), axis=axis, keepdims=keepdims)
    # nanargmax() only has keepdims for numpy>=1.22
    result = np.nanargmax(np.where(condition, value, np.nan), axis=axis)
    if keepdims:
        result = np.expand_dims(result, axis)
    return result


def max_where(
    value: np.ndarray, condition: np.ndarray, axis: int = -1, keepdims: bool = False
) -> np.ndarray:
    assert np.all(np.any(condition, axis=axis)), "require at least one valid element"
    # return np.nanmax(np.where(condition, value, np.nan), axis=axis, keepdims=keepdims)
    result = np.nanmax(np.where(condition, value, np.nan), axis=axis)
    if keepdims:
        result = np.expand_dims(result, axis)
    return result


def arglexmin(keys: Tuple[np.ndarray, ...], axis: int = -1) -> np.ndarray:
    order = np.lexsort(keys, axis=axis)
    return np.take(order, 0, axis=axis)


def arglexmin_where(
    keys: Tuple[np.ndarray, ...],
    condition: np.ndarray,
    axis: int = -1,
    keepdims: bool = False,
) -> np.ndarray:
    assert np.all(np.any(condition, axis=axis)), "require at least one valid element"
    order = np.lexsort(keys, axis=axis)
    # Take first element in order that satisfies condition.
    first_valid = np.expand_dims(
        np.argmax(np.take_along_axis(condition, order, axis=axis), axis=axis), axis
    )
    result = np.take_along_axis(order, first_valid, axis=axis)
    if not keepdims:
        result = np.squeeze(result, axis=axis)
    return result


def argmax_with_confidence(
    value: np.ndarray,
    p: np.ndarray,
    threshold: float,
    condition: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Finds element that maximizes (value, p) subject to p > threshold."""
    mask = p > threshold
    if condition is not None:
        mask = mask & condition
    return arglexmin_where(np.broadcast_arrays(-p, -value), mask)


def plurality_threshold(
    tree: Hierarchy, p: np.ndarray, axis: int = -1, keepdims: bool = False
) -> np.ndarray:
    assert axis in (-1, p.ndim - 1)
    children = tree.children()
    # Find the top 2 elements of each non-trivial family.
    top2_inds = {
        u: inds[np.argsort(p[..., inds], axis=-1)[..., -2:]]
        for u, inds in children.items()
        if len(inds) > 1
    }
    top2_values = np.stack(
        [np.take_along_axis(p, ind, axis=-1) for ind in top2_inds.values()], axis=-1
    )
    # Take the maximum 2nd-best over all non-trivial families.
    threshold = np.max(top2_values, axis=-1)[..., -2]
    if keepdims:
        threshold = np.expand_dims(threshold, -1)
    return threshold


def _apply_to_maybe(fn: Callable, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return fn(x) if x is not None else None


class Sum(nn.Module):
    """Implements sum_xxx as an object. Avoids re-computation."""

    def __init__(
        self,
        tree: Hierarchy,
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
        tree: Hierarchy,
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
        self.prior = torch.from_numpy(uniform_leaf(tree))
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
    tree: Hierarchy, scores: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Returns log-likelihood of each node given its parent."""
    assert dim == -1 or dim == scores.ndim - 1
    num_nodes = tree.num_nodes()
    num_internal = tree.num_internal_nodes()
    node_to_children = tree.children()
    cond_children = [node_to_children[x] for x in tree.internal_subset()]
    cond_num_children = list(map(len, cond_children))
    max_num_children = max(cond_num_children)
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
    tree: Hierarchy, scores: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Returns log-likelihood for conditional softmax."""
    # Finally, take sum over ancestor conditionals to obtain likelihoods.
    assert dim in (-1, scores.ndim - 1)
    log_cond_p = hier_cond_log_softmax(tree, scores, dim=dim)
    device = scores.device
    sum_ancestors_fn = SumAncestors(tree, exclude_root=True).to(device)
    return sum_ancestors_fn(log_cond_p, dim=-1)


class HierCondLogSoftmax(nn.Module):
    """Implements hier_cond_log_softmax as an object. Avoids re-computation."""

    def __init__(self, tree: Hierarchy):
        super().__init__()
        num_nodes = tree.num_nodes()
        num_internal = tree.num_internal_nodes()
        node_to_children = tree.children()
        cond_children = [node_to_children[x] for x in tree.internal_subset()]
        cond_num_children = list(map(len, cond_children))
        max_num_children = max(cond_num_children)
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

    def __init__(self, tree: Hierarchy):
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
    temperature: Optional[float] = None,
) -> torch.Tensor:
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

    def __init__(self, tree: Hierarchy, cut_prob: float, permit_root_cut: bool = False):
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
        drop = torch.bernoulli(
            torch.full((*batch_shape, self.num_nodes), self.cut_prob)
        )
        drop = drop.to(device=device)
        if not self.permit_root_cut:
            drop[..., 0] = 0
        # Check whether to keep all ancestors of each node (drop zero ancestors).
        subtree_mask = self.sum_ancestors(drop, dim=-1) == 0
        # Dilate to keep nodes whose parents belong to subtree.
        subtree_mask = subtree_mask[..., self.node_parent]
        subtree_mask[..., 0] = True  # Root is always kept.
        # Count number of children (do not count parent as child of itself).
        num_children = torch.zeros((*batch_shape, self.num_nodes), device=device)
        num_children = num_children.index_add(
            -1, self.node_parent[1:], subtree_mask[..., 1:].float()
        )
        # Find boundary of subtree.
        boundary = subtree_mask & (num_children == 0)
        return boundary


class RandomCutLoss(nn.Module):
    """Cross-entropy loss using the leaf nodes of a random cut.

    As described in "Deep RTC" (Wu et al., 2020).
    """

    def __init__(
        self,
        tree: Hierarchy,
        cut_prob: float,
        permit_root_cut: bool = False,
        with_leaf_targets: bool = True,
    ):
        super().__init__()
        is_ancestor = tree.ancestor_mask(strict=False)
        # label_to_targets[gt, pr] = 1 iff pr `is_ancestor` gt
        label_to_targets = is_ancestor.T
        if with_leaf_targets:
            label_order = tree.leaf_subset()
            label_to_targets = label_to_targets[label_order]
        else:
            # Need to use FlatSoftmaxNLL?
            raise NotImplementedError

        self.random_cut_fn = RandomCut(
            tree, cut_prob=cut_prob, permit_root_cut=permit_root_cut
        )
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
        cut_targets = cut & targets
        assert torch.all(torch.sum(cut_targets, dim=-1) == 1)
        # loss = F.cross_entropy(cut_scores, cut_targets, reduction='none')
        neg_inf = torch.tensor(-torch.inf, device=scores.device)
        zero = torch.tensor(0.0, device=scores.device)
        pos_score = torch.sum(torch.where(cut_targets, scores, zero), dim=-1)
        loss = -pos_score + torch.logsumexp(torch.where(cut, scores, neg_inf), dim=-1)
        return torch.mean(loss)


class LCAMetric:
    def __init__(self, tree: Hierarchy, value: np.ndarray):
        self.value = value
        self.find_lca = FindLCA(tree)

    def value_at_lca(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        return self.value[lca]

    def value_at_gt(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        gt, _ = np.broadcast_arrays(gt, pr)
        return self.value[gt]

    def value_at_pr(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
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
        with np.errstate(invalid="ignore"):
            return np.where(
                (lca_value == 0) & (gt_value == 0), 1.0, lca_value / gt_value
            )

    def precision(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        pr_value = self.value[pr]
        lca_value = self.value[lca]
        with np.errstate(invalid="ignore"):
            return np.where(
                (lca_value == 0) & (pr_value == 0), 1.0, lca_value / pr_value
            )

    def f1(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        gt_value = self.value[gt]
        pr_value = self.value[pr]
        lca_value = self.value[lca]
        with np.errstate(invalid="ignore"):
            r = np.where((lca_value == 0) & (gt_value == 0), 1.0, lca_value / gt_value)
            p = np.where((lca_value == 0) & (pr_value == 0), 1.0, lca_value / pr_value)
        with np.errstate(divide="ignore"):
            return 2 / (1 / r + 1 / p)


class MarginLoss(nn.Module):
    """Computes soft or hard margin loss for given a margin function."""

    def __init__(
        self,
        tree: Hierarchy,
        with_leaf_targets: bool,
        hardness: str = "soft",
        margin: str = "depth_dist",
        tau: float = 1.0,
    ):
        super().__init__()
        if hardness not in ("soft", "hard"):
            raise ValueError("unknown hardness", hardness)
        n = tree.num_nodes()
        label_order = tree.leaf_subset() if with_leaf_targets else np.arange(n)

        # Construct array label_margin[gt_label, pr_node].
        if margin in ("edge_dist", "depth_dist"):
            # label_margin = metrics.edge_dist(tree, label_order[:, None], np.arange(n)[None, :])
            depth = tree.depths()
            margin_arr = LCAMetric(tree, depth).dist(label_order[:, None], np.arange(n))
        elif margin == "incorrect":
            is_ancestor = tree.ancestor_mask()
            is_correct = is_ancestor[:, label_order].T
            margin_arr = 1 - is_correct
        elif margin == "info_dist":
            info = np.log(tree.num_leaf_nodes() / tree.num_leaf_descendants())
            margin_arr = LCAMetric(tree, info).dist(label_order[:, None], np.arange(n))
        elif margin == "depth_deficient":
            depth = tree.depths()
            margin_arr = LCAMetric(tree, depth).deficient(
                label_order[:, None], np.arange(n)
            )
        elif margin == "log_depth_f1_error":
            depth = tree.depths()
            margin_arr = np.log(
                1 - LCAMetric(tree, depth).f1(label_order[:, None], np.arange(n))
            )
        else:
            raise ValueError("unknown margin", margin)

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
        label_node = (
            labels if self.label_order is None else self.label_order[labels.long()]
        )
        label_score = scores.gather(-1, label_node.unsqueeze(-1)).squeeze(-1)
        label_margin = self.margin[labels.long(), :]
        if self.hardness == "soft":
            loss = -label_score + torch.logsumexp(
                scores + self.tau * label_margin, dim=-1
            )
        elif self.hardness == "hard":
            loss = torch.relu(
                torch.max(
                    scores - label_score.unsqueeze(-1) + self.tau * label_margin,
                    dim=-1,
                )[0]
            )
        else:
            assert False
        return torch.mean(loss)


class FlatSoftmaxNLL(nn.Module):
    """Like cross_entropy() but supports internal labels."""

    def __init__(self, tree, with_leaf_targets: bool = False, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "none", None)
        if with_leaf_targets:
            raise ValueError("use F.cross_entropy() instead!")
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
        logp_ancestors = torch.where(label_leaf_mask, logp_leaf, -inf)
        logp_label = torch.logsumexp(logp_ancestors, dim=-1)
        loss = -logp_label
        if self.reduction == "mean":
            return torch.mean(loss)
        else:
            return loss


def SumLeafDescendants(tree: Hierarchy, **kwargs):
    return SumDescendants(tree, subset=tree.leaf_mask(), **kwargs)
