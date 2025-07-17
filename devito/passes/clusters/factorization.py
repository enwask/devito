from collections import defaultdict
from typing import TypeVar, TYPE_CHECKING

from sympy import Add, Equality, Expr, Mul, S, collect

from devito.ir import cluster_pass
from devito.symbolics import (BasicWrapperMixin, estimate_cost, reuse_if_untouched,
                              retrieve_symbols, q_routine)
from devito.tools import ReducerMap
from devito.tools.visitors import dag_visitor
from devito.types.object import AbstractObject

__all__ = ['factorize']


MIN_COST_FACTORIZE = 100
"""
Minimum operation count of an expression so that aggressive factorization
is applied.
"""


@cluster_pass
def factorize(cluster, *args, options=None, **kwargs):
    """
    Factorize trascendental functions, symbolic powers, numeric coefficients.

    If the expression has an operation count greater than ``MIN_COST_FACTORIZE``,
    then the algorithm is applied recursively until no more factorization
    opportunities are detected.
    """
    try:
        strategy = options['fact-schedule']
    except TypeError:
        strategy = 'basic'

    handle = collect_nested(cluster.exprs, strategy)
    costs_handle = list(map(estimate_cost, handle))

    if any(cost >= MIN_COST_FACTORIZE for cost in costs_handle):
        handle_prev, costs_prev = handle, list(map(estimate_cost, cluster.exprs))
        while any(cost < cost_prev for cost, cost_prev in zip(costs_handle, costs_prev)):
            handle_prev, handle = handle, collect_nested(handle, strategy)
            costs_prev, costs_handle = costs_handle, list(map(estimate_cost, handle))
        costs_handle, handle = costs_prev, handle_prev

    return cluster.rebuild(handle)


def collect_special(expr, strategy):
    """
    Factorize elemental functions, pows, and other special symbolic objects,
    prioritizing the most expensive entities.
    """
    args, candidates = zip(*[_collect_nested(a, strategy) for a in expr.args])
    candidates = ReducerMap.fromdicts(*candidates)

    funcs = candidates.getall('funcs', [])
    pows = candidates.getall('pows', [])
    coeffs = candidates.getall('coeffs', [])

    # Functions/Pows are collected first, coefficients afterwards
    terms = []
    w_funcs = []
    w_pows = []
    w_coeffs = []
    for i in args:
        _args = i.args
        if any(j in funcs for j in _args):
            w_funcs.append(i)
        elif any(j in pows for j in _args):
            w_pows.append(i)
        elif any(j in coeffs for j in _args):
            w_coeffs.append(i)
        else:
            terms.append(i)

    # Any factorization possible?
    if not any(len(i) > 1 for i in [w_funcs, w_pows, w_coeffs]):
        # `evaluate=True` below guarantees de-nesting of Adds
        # For example:
        # `args[0] = ((0.1*(a + b)) + (0.2*(c + d)))`
        # `args[1] = ((0.1*(e + f)) + (0.2*(g + h)))`
        # ->
        # `expr = (0.1*(a + b) + 0.2*(c + d) + 0.1*(e + f) + 0.2*(g + h))`
        # Which is then further factorizable by `collect_const`
        return reuse_if_untouched(expr, args, evaluate=True)

    # Collect common funcs
    if len(w_funcs) > 1:
        w_funcs = Add(*w_funcs, evaluate=False)
        w_funcs = collect(w_funcs, funcs, evaluate=False)
        try:
            terms.extend([Mul(k, collect_const(v), evaluate=False)
                          for k, v in w_funcs.items()])
        except AttributeError:
            assert w_funcs == 0
    else:
        terms.extend(w_funcs)

    # Collect common pows
    w_pows = Add(*w_pows, evaluate=False)
    w_pows = collect(w_pows, pows, evaluate=False)
    try:
        terms.extend([Mul(k, collect_const(v), evaluate=False)
                      for k, v in w_pows.items()])
    except AttributeError:
        assert w_pows == 0

    # Collect common temporaries (r0, r1, ...)
    w_coeffs = Add(*w_coeffs, evaluate=False)
    symbols = retrieve_symbols(w_coeffs)
    if len(set(symbols)) != len(symbols):
        w_coeffs = collect(w_coeffs, symbols, evaluate=False)
        try:
            terms.extend([Mul(k, collect_const(v), evaluate=False)
                          for k, v in w_coeffs.items()])
        except AttributeError:
            assert w_coeffs == 0
    else:
        terms.append(w_coeffs)

    return Add(*terms)


def collect_const(expr):
    """
    *Much* faster alternative to sympy.collect_const. *Potentially* slightly less
    powerful with complex expressions, but it is equally effective for the
    expressions we have to deal with.
    """
    # Running example: `a*3. + b*2. + c*3.`

    # -> {a: 3., b: 2., c: 3.}
    mapper = expr.as_coefficients_dict()

    # -> {3.: [a, c], 2: [b]}
    inverse_mapper = defaultdict(list)
    for k, v in mapper.items():
        if v >= 0:
            inverse_mapper[v].append(k)
        else:
            inverse_mapper[-v].append(-k)

    # Any factorization possible?
    if len(inverse_mapper) == len(expr.args) or \
       (len(inverse_mapper) == 1 and 1 in inverse_mapper):
        return expr

    terms = []
    for k, v in inverse_mapper.items():
        if len(v) == 1 and not v[0].is_Add:
            # Special case: avoid e.g. (-2)*a
            mul = Mul(k, *v)
        elif all(i.is_Mul and i.args[0] == -1 for i in v):
            # Other special case: [-a, -b, -c ...]
            operands = [Mul(*i.args[1:], evaluate=False) for i in v]
            add = Add(*operands, evaluate=False)
            mul = Mul(-k, add, evaluate=False)
        elif k == 1:
            # 1 * (a + c)
            mul = Add(*v)
        else:
            # Back to the running example
            # -> (a + c)
            add = Add(*v)
            if add == 0:
                mul = S.Zero
            else:
                # -> 3.*(a + c)
                mul = Mul(k, add, evaluate=False)

        terms.append(mul)

    return Add(*terms)


def strategy0(expr, strategy):
    rebuilt = collect_special(expr, strategy)
    rebuilt = collect_const(rebuilt)

    return rebuilt


strategies = {
    'basic': strategy0
}


# Describes the type being visited by _collect_nested
ExprType = TypeVar('ExprType', bound=Expr | tuple[Expr])


@dag_visitor()
def _collect_nested(maybe_exprs: ExprType, strategy: str) \
        -> tuple[ExprType, dict | ReducerMap]:
    """
    Recursion helper for `collect_nested`.
    """
    if isinstance(maybe_exprs, tuple):
        exprs, candidates = zip(*[_collect_nested(i, strategy) for i in maybe_exprs])
        return tuple(exprs), ReducerMap.fromdicts(*candidates)

    # Not a tuple
    expr = maybe_exprs

    # Return semantic (rebuilt expression, factorization candidates)
    if expr.is_Number:
        return expr, {'coeffs': expr}

    if q_routine(expr):
        # E.g., a DefFunction
        args, candidates = zip(*[_collect_nested(a, strategy) for a in expr.args])
        return expr.func(*args, evaluate=False), {}

    if expr.is_Function:
        return expr, {'funcs': expr}

    if expr.is_Pow:
        return expr, {'pows': expr}

    if (expr.is_Symbol or expr.is_Indexed or not expr.args or
          isinstance(expr, (BasicWrapperMixin, AbstractObject))):
        return expr, {}

    if expr.is_Add:
        return strategies[strategy](expr, strategy), {}

    if expr.is_Mul:
        args, candidates = zip(*[_collect_nested(a, strategy) for a in expr.args])
        expr = reuse_if_untouched(expr, args, evaluate=True)
        return expr, ReducerMap.fromdicts(*candidates)

    if expr.is_Equality:
        if TYPE_CHECKING:
            assert isinstance(expr, Equality)

        rhs, _ = _collect_nested(expr.rhs, strategy)
        expr = reuse_if_untouched(expr, (expr.lhs, rhs))
        return expr, {}

    args, candidates = zip(*[_collect_nested(a, strategy) for a in expr.args])
    return expr.func(*args), ReducerMap.fromdicts(*candidates)


def collect_nested(exprs: tuple[Expr], strategy='basic') -> tuple[Expr]:
    """
    Collect numeric coefficients, trascendental functions, pows, and other
    symbolic objects across all levels of the expression tree.

    Parameters
    ----------
    expr : expr-like
        The expression to be factorized.
    """
    return _collect_nested.visit(exprs, strategy)[0]
