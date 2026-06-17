import itertools
from operator import itemgetter
from typing import (
    Iterable,
    Any,
    Callable,
    Sequence,
)
from types import GeneratorType
from collections import deque

from sage.graphs.digraph import DiGraph

type NestedTuple[T] = None | tuple[T, NestedTuple[T]]


def dedupe_adjacent[T](iterable: Iterable[T]) -> Iterable[T]:
    return map(itemgetter(0), itertools.groupby(iterable))


def grouped_by[T, K](iterable: Iterable[T], f: Callable[[T], K]) -> dict[K, list[T]]:
    result = dict[K, list[T]]()
    for item in iterable:
        if (k := f(item)) in result:
            result[k].append(item)
        else:
            result[k] = [item]

    return result


def minmax[T](iterable: Iterable[T]):
    iterator = iter(iterable)
    try:
        min = max = next(iterator)
    except StopIteration:
        raise ValueError("minmax() iterable argument is empty") from None
    for item in iterator:
        if item < min:
            min = item
        if item > max:
            max = item

    return min, max


def echo(*args):
    """Print arguments and return the last one"""
    print(*args)
    return args[-1]


def with_attributes(obj, **kwargs):
    for k, v in kwargs.items():
        setattr(obj, k, v)

    return obj


def countlen[T](iterable: Iterable[T]):
    return sum((1 for _ in iterable), start=0)


def rle[T](iterable: Iterable[T]):
    """
    Generate the run length encoding of an iteratble

    Yields pairs of the form (element, count)
    """
    return ((x, countlen(xs)) for x, xs in itertools.groupby(iterable))


def from_rle[T](iterable: Iterable[tuple[T, int]]):
    """Yield elements of a run length encoded sequence"""
    return (x for x, n in iterable for _ in range(n))


def sgn(x):
    return -1 if x < 0 else 0 if x == 0 else 1


_MISSING = object()


def first[T, DT](iterable: Iterable[T], default: DT = _MISSING):
    try:
        return next(iter(iterable))
    except StopIteration:
        if default is _MISSING:
            raise (ValueError("Received empty iterable with no default specified"))
        else:
            return default


def run_positions(A: Iterable):
    """
    Return indices of maximal runs of identical elements in a sequence

    Input: An iterable ``A``

    Output: A generator yielding pairs ``(ai, bi)`` corresponding to maximal constant slices ``A[ai:bi]``.
    """
    start = i = 0
    for x, y in itertools.pairwise(A):
        i += 1
        if x != y:
            yield start, i
            start = i
    yield start, i + 1


def inverse_path(path: Sequence[int]):
    return tuple(s for s in reversed(path))


def to_turn[T](u: T | None, v: T | None) -> tuple[T, T] | tuple[None, T | None]:
    return (
        (u, v) if u is None else (v, u) if v is None else (u, v) if u <= v else (v, u)
    )


def first_truthy[T, DT](iterable: Iterable[T], default: DT = False, key=None):
    return next(filter(key, iterable), default)


def stack_graph(stacks, images, stack_index=None, canonicalize=False) -> DiGraph:
    if stack_index is None:
        stack_index = {}
        for i, stack in enumerate(stacks):
            for e in stack:
                stack_index[e] = i

    graph = DiGraph(len(stacks), multiedges=True, loops=True)

    for e, img in images:
        i = stack_index[e]
        for img_e in img:
            if img_e is not None:
                graph.add_edge(i, stack_index[abs(img_e)])

    if canonicalize:
        graph, perm = graph.canonical_label(algorithm="sage", certificate=True)

        for v in stack_index.keys():
            stack_index[v] = perm[stack_index[v]]

        d: dict[int, int] = {v: k for k, v in perm.items() if k != v}
        while d:
            k, v = d.popitem()
            tmp_s = stacks[v]
            tmp_e = images[v]
            while v in d:
                nv = d.pop(v)
                stacks[v] = stacks[nv]
                images[v] = images[nv]
                v = nv
            stacks[k] = tmp_s
            images[k] = tmp_e

    return graph


def transpose_list[T](iterables: Iterable[Iterable[T]]):
    iterators = [iter(it) for it in iterables]
    while iterators:
        try:
            yield [next(it) for it in iterators]
        except StopIteration:
            return


def unnest[T](coll: NestedTuple[T]):
    while True:
        match coll:
            case (x, coll):
                yield x
            case _:
                return


def consume(iterator, n=None):
    """Advance the iterator n-steps ahead. If n is None, consume entirely."""
    # Use functions that consume iterators at C speed.
    if n is None:
        deque(iterator, maxlen=0)
    else:
        next(itertools.islice(iterator, n, n), None)


def nneg(x):
    return 0 if x < 0 else x


def shortlex(x):
    return len(x), x


def lsbi(n):
    """Return index of lowest set bit, or -1 if `n` is 0"""
    return (n & -n).bit_length() - 1


def deep_tuple(coll: Iterable) -> tuple[Any, ...]:
    return (
        *(
            deep_tuple(x) if isinstance(x, list | tuple | GeneratorType) else x
            for x in coll
        ),
    )
