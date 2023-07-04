import collections
import csv
import itertools
from typing import Callable, Dict, Hashable, List, Optional, Sequence, TextIO, Tuple

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike


class Hierarchy:
    """Hierarchy of nodes 0, ..., n-1."""

    def __init__(self, parents):
        n = len(parents)
        assert np.all(parents[1:] < np.arange(1, n))
        self._parents = parents

    def num_nodes(self) -> int:
        return len(self._parents)

    def edges(self) -> List[Tuple[int, int]]:
        return list(zip(self._parents[1:], itertools.count(1)))

    def parents(self, root_loop: bool = False) -> np.ndarray:
        if root_loop:
            return np.where(
                self._parents >= 0, self._parents, np.arange(len(self._parents))
            )
        else:
            return np.array(self._parents)

    def children(self) -> Dict[int, np.ndarray]:
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
        # n = len(self._parents)
        # d = np.zeros([n], dtype=int)
        # for i, j in self.edges():
        #     assert i < j, 'require edges in topological order'
        #     d[j] = d[i] + 1
        # return d
        return self.accumulate_ancestors(np.add, (self._parents >= 0).astype(int))

    def num_leaf_descendants(self) -> np.ndarray:
        # c = self.leaf_mask().astype(int)
        # for i, j in reversed(self.edges()):
        #     assert i < j, 'require edges in topological order'
        #     c[i] += c[j]
        # return c
        return self.accumulate_descendants(np.add, self.leaf_mask().astype(int))

    def max_heights(self) -> np.ndarray:
        heights = np.zeros_like(self.depths())
        # for i, j in reversed(self.edges()):
        #     heights[i] = max(heights[i], heights[j] + 1)
        # return heights
        return self.accumulate_descendants(lambda u, v: max(u, v + 1), heights)

    def min_heights(self) -> np.ndarray:
        # Initialize leaf nodes to zero, internal nodes to upper bound.
        #   height + depth <= max_depth
        #   height <= max_depth - depth
        depths = self.depths()
        heights = np.where(self.leaf_mask(), 0, depths.max() - depths)
        # for i, j in reversed(self.edges()):
        #     heights[i] = min(heights[i], heights[j] + 1)
        # return heights
        return self.accumulate_descendants(lambda u, v: min(u, v + 1), heights)

    # def accumulate_ancestors_inplace(self, func: Callable, values: MutableSequence):
    #     # Start from root and move down.
    #     for i, j in self.edges():
    #         values[j] = func(values[i], values[j])

    # def accumulate_descendants_inplace(self, func: Callable, values: MutableSequence):
    #     # Start from leaves and move up.
    #     for i, j in reversed(self.edges()):
    #         values[i] = func(values[i], values[j])

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
    ) -> List[np.ndarray]:
        # TODO: Could avoid potential high memory usage here using parents.
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

    def __str__(self, node_names: Optional[List[str]] = None) -> str:
        return format_tree(self, node_names)

    def to_networkx(self, keys: Optional[List] = None) -> nx.DiGraph:
        n = self.num_nodes()
        g = nx.DiGraph()
        if keys is None:
            keys = list(range(n))
        g.add_edges_from([(keys[i], keys[j]) for i, j in self.edges()])
        return g


def make_hierarchy_from_edges(
    pairs: Sequence[Tuple[str, str]],
) -> Tuple[Hierarchy, List[str]]:
    """Creates a hierarchy from a list of name pairs.

    The order of the edges determines the order of the nodes.
    (Each node except the root appears once as a child.)
    The root is placed first in the order.
    """
    num_edges = len(pairs)
    num_nodes = num_edges + 1
    # Data structures to populate from list of pairs.
    parents = np.full([num_nodes], -1, dtype=int)
    names = [None] * num_nodes
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
    return Hierarchy(parents), names


def load_edges(f: TextIO, delimiter=",") -> List[Tuple[str, str]]:
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
) -> Tuple[Hierarchy, np.ndarray]:
    nodes = ancestors_union(tree, nodes)
    subtree = rooted_subtree(tree, nodes)
    return subtree, nodes


def ancestors_union(tree: Hierarchy, node_subset: np.ndarray) -> np.ndarray:
    """Returns union of ancestors of nodes."""
    paths = tree.paths_padded(-1)
    paths = paths[node_subset]
    return np.unique(paths[paths >= 0])


def find_subset_index(base: List[Hashable], subset: List[Hashable]) -> np.ndarray:
    """Returns index of subset elements in base list (injective map)."""
    name_to_index = {x: i for i, x in enumerate(base)}
    return np.asarray([name_to_index[x] for x in subset], dtype=int)


def find_projection(tree: Hierarchy, node_subset: np.ndarray) -> np.ndarray:
    """Finds projection to nearest ancestor in subtree."""
    # TODO: Only works for rooted sub-trees?
    # Use paths rather than ancestor_mask to avoid large memory usage.
    assert np.all(node_subset >= 0)
    paths = tree.paths_padded(-1)
    # Find index in subset.
    reindex = np.full([tree.num_nodes()], -1)
    reindex[node_subset] = np.arange(len(node_subset))
    subset_paths = np.where(paths >= 0, reindex[paths], -1)
    deepest = _last_nonzero(subset_paths >= 0, axis=1)
    # Check that all ancestors are present.
    # TODO: Could consider removing for non-rooted sub-trees?
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
    # TODO: Ensure exact.
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
    tree: Hierarchy, node_names: Optional[List[str]] = None, include_size: bool = False
) -> str:
    if node_names is None:
        node_names = [str(i) for i in range(tree.num_nodes())]

    node_to_children = tree.children()
    node_sizes = tree.num_leaf_descendants()

    # def subtree(node, prefix, is_last):
    #     yield prefix + ('└── ' if is_last else '├── ') + node_names[node] + '\n'
    #     children = node_to_children.get(node, ())
    #     child_prefix = prefix + ('    ' if is_last else '│   ')
    #     for i, child in enumerate(children):
    #         child_is_last = (i == len(children) - 1)
    #         yield from subtree(child, child_prefix, child_is_last)

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


def level_nodes(tree: Hierarchy, extend: bool = False) -> List[np.ndarray]:
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
) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
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
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Returns the list of parents and their children at each depth."""
    level_parents, level_children = level_successors(tree)
    level_children = [
        _pad_2d(x, dtype=int, method=method, constant_value=constant_value)
        for x in level_children
    ]
    return level_parents, level_children


def siblings(tree: Hierarchy) -> List[np.ndarray]:
    node_parent = tree.parents()
    node_children = tree.children()
    node_children[-1] = np.empty([0], dtype=int)
    return [_except(u, node_children[node_parent[u]]) for u in range(tree.num_nodes())]


def siblings_padded(
    tree: Hierarchy, method: str = "constant", constant_value: int = -1
) -> np.ndarray:
    ragged = siblings(tree)
    return _pad_2d(ragged, dtype=int, method=method, constant_value=constant_value)


def _except(v, x: np.ndarray) -> np.ndarray:
    return x[x != v]


def _pad_2d(
    x: List[np.ndarray], dtype: np.dtype, method: str = "constant", constant_value=None
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
