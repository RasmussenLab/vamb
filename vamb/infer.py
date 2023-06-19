from typing import Optional, Tuple

import numpy as np

import vamb.hier as hier


def most_confident_leaf(tree: hier.Hierarchy, p: np.ndarray) -> np.ndarray:
    assert p.shape[-1] == tree.num_nodes()
    is_leaf = tree.leaf_mask()
    return argmax_where(p, is_leaf)


def max_info_majority_subtree(tree: hier.Hierarchy, p: np.ndarray) -> np.ndarray:
    assert p.shape[-1] == tree.num_nodes()
    # -x is a monotonic mapping of -log(1/(x/n)) = log x - constant
    specificity = -tree.num_leaf_descendants()
    # Trivial nodes are allowed in the tree, but we will not predict them.
    # If the parent is equal to the child, it has the same score and specificity.
    # Maybe we should break ties in (specificity, p) using depth?
    not_trivial = (tree.num_children() != 1)
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
    assert np.any(is_valid), 'require at least one valid element'
    prob = prob[is_valid]
    info = info[is_valid]
    valid_inds, = np.nonzero(is_valid)

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
        if np.any((prob[1:] == prob[:-1]) & (info[1:] == info[:-1]) & (keep[1:] | keep[:-1])):
            raise ValueError('set is not unique')

    return valid_inds[order[keep]]


def argmax_where(
        value: np.ndarray,
        condition: np.ndarray,
        axis: int = -1,
        keepdims: bool = False) -> np.ndarray:
    # Will raise an exception if not np.all(np.any(condition, axis=axis)).
    # return np.nanargmax(np.where(condition, value, np.nan), axis=axis, keepdims=keepdims)
    # nanargmax() only has keepdims for numpy>=1.22
    result = np.nanargmax(np.where(condition, value, np.nan), axis=axis)
    if keepdims:
        result = np.expand_dims(result, axis)
    return result


def max_where(
        value: np.ndarray,
        condition: np.ndarray,
        axis: int = -1,
        keepdims: bool = False) -> np.ndarray:
    assert np.all(np.any(condition, axis=axis)), 'require at least one valid element'
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
    # TODO: Make more efficient (linear rather than log-linear).
    assert np.all(np.any(condition, axis=axis)), 'require at least one valid element'
    order = np.lexsort(keys, axis=axis)
    # Take first element in order that satisfies condition.
    # TODO: Would be faster to take subset and then sort?
    # Would this break the vectorization?
    # first_valid = np.argmax(np.take_along_axis(condition, order, axis=axis),
    #                         axis=axis, keepdims=True)
    first_valid = np.expand_dims(
        np.argmax(np.take_along_axis(condition, order, axis=axis), axis=axis),
        axis)
    result = np.take_along_axis(order, first_valid, axis=axis)
    if not keepdims:
        result = np.squeeze(result, axis=axis)
    return result


def argmax_with_confidence(
        value: np.ndarray,
        p: np.ndarray,
        threshold: float,
        condition: Optional[np.ndarray] = None) -> np.ndarray:
    """Finds element that maximizes (value, p) subject to p > threshold."""
    mask = (p > threshold)
    if condition is not None:
        mask = mask & condition
    return arglexmin_where(np.broadcast_arrays(-p, -value), mask)


def plurality_threshold(
        tree: hier.Hierarchy,
        p: np.ndarray,
        axis: int = -1,
        keepdims: bool = False) -> np.ndarray:
    assert axis in (-1, p.ndim - 1)
    children = tree.children()
    # Find the top 2 elements of each non-trivial family.
    top2_inds = {
        u: inds[np.argsort(p[..., inds], axis=-1)[..., -2:]]
        for u, inds in children.items() if len(inds) > 1
    }
    top2_values = np.stack([
        np.take_along_axis(p, ind, axis=-1)
        for ind in top2_inds.values()
    ], axis=-1)
    # Take the maximum 2nd-best over all non-trivial families.
    threshold = np.max(top2_values, axis=-1)[..., -2]
    if keepdims:
        threshold = np.expand_dims(threshold, -1)
    return threshold