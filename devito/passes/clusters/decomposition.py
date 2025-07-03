from collections.abc import Iterable, Iterator
from functools import singledispatch
from typing import NamedTuple
from sympy import Expr, Indexed, Symbol

from devito.symbolics.manipulation import _uxreplace
from devito.tools.data_structures import DAG, frozendict
from devito.types.misc import Temp
try:
    from sympy.core.core import ordering_of_classes
except ImportError:
    # Moved in 1.13
    from sympy.core.basic import ordering_of_classes

import numpy as np
import sympy

from devito.finite_differences.differentiable import IndexDerivative
from devito.ir.clusters.cluster import Cluster
from devito.ir.clusters.visitors import cluster_pass
from devito.ir.iet.nodes import Node
from devito.ir.support.symregistry import SymbolRegistry
from devito.tools.dtypes_lowering import extract_dtype
from devito.tools.utils import as_list


__all__ = ['decompose_clusters']


def extract_conditionals(expr: Expr) -> frozendict:
    try:
        return expr.conditionals
    except AttributeError:
        return frozendict()


class Candidate:
    """
    A candidate expression with associated conditionals and metadata.
    """

    __slots__ = ('expr', 'conditionals', 'depth', 'subtree_size')

    def __init__(self, expr: Expr, depth: int,
                 conditionals: frozendict | None = None) -> None:
        self.expr = expr
        self.depth = depth
        self.subtree_size = 1
        self.conditionals = conditionals or extract_conditionals(expr)

    def key(self) -> tuple[Expr, frozendict]:
        """
        Returns a key for mapping this candidate to a temporary.
        """
        return self.expr, self.conditionals


class CSubtreeTemp(Temp):
    """
    A cluster-level Temp extracted by cluster decomposition, ignored by CSE's
    dropping of temporaries.
    """
    ordering_of_classes.insert(ordering_of_classes.index('Temp') + 1, 'CSubtreeTemp')


@cluster_pass
def decompose_clusters(cluster: Cluster, sregistry: SymbolRegistry = None, options = None, **kwargs) -> Cluster:
    """
    Heuristically decomposes the expression tree into smaller subtrees to avoid
    deeply nested expressions, which can lead to poor performance and memory usage.
    """
    # Min dtype for temporaries
    try:
        min_dtype = np.promote_types(options['scalar-min-type'], cluster.dtype).type
    except TypeError:
        min_dtype = cluster.dtype

    # Get a list of expressions in the cluster (or accept expressions directly)
    exprs = as_list(cluster.exprs)

    # Choose expressions to extract into temporaries based on heuristics
    candidates = list(_find_candidates(exprs))
    targets = pick_temporaries(candidates)

    # Map expressions to temporaries
    temps: dict[tuple[Expr, frozendict], CSubtreeTemp] = {}
    for candidate in targets:
        # Create a temporary to replace this symbol with
        key = candidate.key()
        dtype = np.promote_types(extract_dtype(candidate.expr), min_dtype).type
        temps[key] = CSubtreeTemp(name=sregistry.make_name(prefix='d'), dtype=dtype)

    processed: list[Expr] = []
    temps_assigned: set[tuple[Expr, frozendict]] = set()

    # Replace occurrences in expressions
    for expr in exprs:
        conditionals = extract_conditionals(expr)
        for key, temp in temps.items():
            temp_expr, temp_cond = key
            if conditionals != temp_cond:
                # Don't apply replacements if conditionals differ
                continue

            # Apply substitution for this temporary
            expr, flag = _uxreplace(expr, {temp_expr: temp})

            if flag and not key in temps_assigned:
                # Assign a value to the temporary if we haven't yet
                processed.append(expr.func(temp, temp_expr, operation=None))
                temps_assigned.add(key)

        # Done processing this expression
        processed.append(expr)

    # If anything changed, ensure toplogical order
    if temps_assigned:
        processed = _toposort(processed)

    return cluster.rebuild(exprs=processed)


def compute_cost_heuristic(candidate: Candidate) -> float:
    """
    Calculates a heuristic cost based on the depth of an expression
    and the size of its subtree.
    """
    # FIXME: This is just some arbitrary bullshit for now
    if candidate.subtree_size == 1:
        return -1.0  # No reason to extract leaves

    return candidate.depth ** 2 + candidate.subtree_size


def pick_temporaries(candidates: list[Candidate]) -> list[Candidate]:
    """
    Picks nodes to extract into temporaries.
    """
    if not candidates:
        return []

    # FIXME: More arbitrary bullshit
    # We pick nodes whose costs deviate significantly from the mean
    costs = list(map(compute_cost_heuristic, candidates))
    mean, std = np.mean(costs), np.std(costs)
    threshold = mean + 1 * std

    # Make unique
    return list({cand for cand, cost in zip(candidates, costs)
                 if cost >= threshold and cand.subtree_size > 1})


def _toposort(exprs: list[Expr]) -> list[Expr]:
    """
    Ensures topological order of expressions.
    """
    # FIXME: For now just duplicating code from CSE, but there should be a better way
    dag = DAG(exprs)

    for e0 in exprs:
        if not isinstance(e0.lhs, CSubtreeTemp):
            continue

        for e1 in exprs:
            if e0.lhs in e1.rhs.free_symbols:
                dag.add_edge(e0, e1, force_add=True)

    def choose_element(queue, scheduled):
        tmps = [i for i in queue if isinstance(i.lhs, CSubtreeTemp)]
        if tmps:
            # Try to honor temporary names as much as possible
            first = sorted(tmps, key=lambda i: i.lhs.name).pop(0)
            queue.remove(first)
        else:
            first = sorted(queue, key=lambda i: exprs.index(i)).pop(0)
            queue.remove(first)
        return first

    processed = dag.topological_sort(choose_element)

    return processed


@singledispatch
def _find_candidates(obj: object, depth: int = 0,
                     conditionals: frozendict | None = None) -> Iterator[Candidate]:
    """
    Finds candidate expressions for extraction into temporaries.

    Expression nodes get a computed depth and subtree size. The depth counts traversal
    through iterables and other intermediaries; subtree size counts only the number of
    descendant expression nodes (i.e. the number of candidates in the subtree).
    """
    yield from ()


@_find_candidates.register(tuple)
@_find_candidates.register(list)
def _(exprs: Iterable[Expr], depth: int = 0,
      conditionals: frozendict | None = None) -> Iterator[Candidate]:
    for node in exprs:
        yield from _find_candidates(node, depth=depth + 1, conditionals=conditionals)


@_find_candidates.register(sympy.Eq)
def _(expr: sympy.Eq, depth: int = 0,
      conditionals: frozendict | None = None) -> Iterator[Candidate]:
    yield from _find_candidates(expr.rhs, depth=depth + 1,
                                conditionals=extract_conditionals(expr))


@_find_candidates.register(Indexed)
@_find_candidates.register(Symbol)
def _(expr: Expr, depth: int = 0,
      conditionals: frozendict | None = None) -> Iterator[Candidate]:
    # Don't enter indexeds or symbols
    yield from ()


@_find_candidates.register(IndexDerivative)
def _(expr: Expr, depth: int = 0,
      conditionals: frozendict | None = None) -> Iterator[Candidate]:
    # Don't enter IndexDerivatives, but possibly extract them
    yield Candidate(expr=expr, depth=depth, conditionals=conditionals)


@_find_candidates.register(Expr)
def _(expr: Expr, depth: int = 0,
      conditionals: frozendict | None = None) -> Iterator[Candidate]:
    # Candidate for the root of this subtree
    root = Candidate(expr=expr, depth=depth, conditionals=conditionals)

    # tuple visitor will increment depth for us
    for candidate in _find_candidates(expr.args, depth=depth,
                                      conditionals=conditionals):
        # Update subtree size and propagate this candidate
        root.subtree_size += candidate.subtree_size
        yield candidate

    yield root
