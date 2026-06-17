import math
from itertools import pairwise, product, islice
from typing import Generator, Iterable

import numpy as np
from sage.combinat.integer_vectors_mod_permgroup import (
    IntegerVectorsModPermutationGroup_with_constraints,
)
from sage.graphs.digraph import DiGraph
from sage.graphs.graph import Graph
from sage.graphs.tutte_polynomial import edge_multiplicities
from sage.matrix.constructor import matrix
from sage.rings.qqbar import AlgebraicNumber
from scipy.sparse.linalg import eigs

from utils import dedupe_adjacent, grouped_by

type CycleWithMultiplicity = tuple[list[int], int]
type BridgePair = tuple[int | tuple[int, int], int | tuple[int, int]]
type Orbits = tuple[tuple[int, ...], ...]


def simple_cycles(dg: DiGraph) -> Generator[list]:
    def simple_cycles_through_edge(g: DiGraph, u, v):
        # get rid of unreachable vertices
        g.subgraph(
            g.connected_component_containing_vertex(u),
            inplace=True,
            immutable=False,
        )
        g.delete_edge(u, v)
        for path in g.shortest_simple_paths(source=v, target=u):
            path.append(v)
            yield path

    # to_simple gets called in shortest_simple_path if graph has loops or multiedges
    # so we should call it once at the beginning for efficiency
    components = dg.to_simple(
        to_undirected=False,
        immutable=False,
    ).strongly_connected_components_subgraphs()

    for v in dg.loop_vertices():
        yield [v, v]

    while components:
        g = components.pop()
        if not g.size():
            continue
        u, v = next(g.edge_iterator(labels=False))
        yield from simple_cycles_through_edge(g, u, v)
        components.extend(g.strongly_connected_components_subgraphs())


def simple_cycles_with_multiplicity(g: DiGraph) -> Generator[CycleWithMultiplicity]:
    edge_mults = edge_multiplicities(g)
    for cycle in simple_cycles(g):
        yield cycle, math.prod(edge_mults[e] for e in pairwise(cycle))


def simple_closed_curves(g: DiGraph):
    for cycle in simple_cycles(g):
        yield from product(
            *(tuple((u, v, l) for l in g.edge_label(u, v)) for u, v in pairwise(cycle))
        )


def simple_closed_curve_count(g: DiGraph):
    return sum(m for _, m in simple_cycles_with_multiplicity(g))


def curve_complex(data: DiGraph | Iterable[CycleWithMultiplicity], canonical=True):
    cycles = (
        [*simple_cycles_with_multiplicity(data)]
        if isinstance(data, DiGraph)
        else [*data]
    )
    cc = Graph(sum(m for _, m in cycles), multiedges=False, loops=False)
    i_offset = 0
    for i, (ci, mi) in enumerate(cycles):
        ci = {*ci}
        j_offset = 0
        for j, (cj, mj) in enumerate(islice(cycles, i)):
            if ci.isdisjoint(cj):
                cc.add_edges(
                    product(
                        range(i_offset, i_offset + mi),
                        range(j_offset, j_offset + mj),
                    )
                )
            j_offset += mj
        i_offset += mi
    if canonical and cc.n_edges:
        return cc.canonical_label(algorithm="sage", immutable=True)
    else:
        return cc


named_curve_complexes = {
    name: Graph(spec).canonical_label(immutable=True)
    for name, spec in [
        ("A1", 1),
        ("2A1", 2),
        ("3A1", 3),
        ("4A1", 4),
        ("5A1", 5),
        ("6A1", 6),
        ("7A1", 7),
        ("8A1", 8),
        ("A2", [[0, 1], [(0, 1)]]),
        ("A2*", [[0, 1, 2], [(0, 1)]]),
        ("A2**", [[0, 1, 2], [(0, 1)]]),
    ]
}


def integer_vectors_inplace(n: int, k: int):
    """
    A fast iterator for integer vectors of ``n`` of length ``k``

    yields a list of integers which is modified in-place upon future iterations.
    The output should be treated as read-only.
    """
    if n < 0 or k < 0:
        return

    if not k:
        if not n:
            yield []
        return
    if k == 1:
        yield [n]
        return

    pos = 0  # Current position
    rem = 0  # Amount remaining
    cur = [0] * k  # current list
    cur[0] = n
    yield cur
    while pos >= 0:
        if not cur[pos]:
            pos -= 1
            continue
        cur[pos] -= 1
        rem += 1
        if not rem:
            yield cur
        elif pos == k - 2:
            cur[pos + 1] = rem
            yield cur
            cur[pos + 1] = 0
        else:
            pos += 1
            cur[pos] = rem  # Guaranteed to be at least 1
            rem = 0
            yield cur


def subdivide_edge(g: DiGraph, u, v):
    # l = g.edge_label(u, v)[0]
    w = g.add_vertex()
    g.delete_edge(u, v, None)
    g.add_edge(u, w, None)
    g.add_edge(w, v, None)
    return w


def bridge(g: DiGraph, a, b):
    g = g.copy(immutable=False)
    match a:
        case (v, w):
            if a == b:
                a = b = subdivide_edge(g, v, w)
            else:
                a = subdivide_edge(g, v, w)
    match b:
        case (v, w):
            b = subdivide_edge(g, v, w)

    g.add_edge(a, b)

    return g.canonical_label(algorithm="sage", immutable=True)


def bridge_pairs(g: DiGraph):
    verts = g.vertices()
    edges = [*dedupe_adjacent(g.edge_iterator(labels=False))]
    for u in verts:
        for v in verts:
            yield u, v

        for e in edges:
            yield u, e

    for e in edges:
        for v in verts:
            yield e, v
        for f in edges:
            if e == f and e[0] == e[1] and g.degree(e[0]) == 2:
                continue
            yield e, f


def topological_digraphs_with_curve_complexes(
    ccs: Graph | Iterable[Graph],
) -> list[set[DiGraph]]:
    # target_curve_counts = grouped_by(ccs, len)
    if isinstance(ccs, Graph):
        ccs = (ccs,)
    indices = {g.canonical_label(immutable=True): i for i, g in enumerate(ccs)}
    targets = grouped_by(indices.keys(), len)
    result = [set[DiGraph]() for _ in indices]
    max_curve_count = max(targets)

    descendants = {
        DiGraph(
            [[0], [(0, 0)]],
            format="vertices_and_edges",
            multiedges=True,
            loops=True,
        ).canonical_label(algorithm="sage", immutable=True),
    }

    while gs := descendants:
        descendants = set[DiGraph]()
        for g in gs:
            cycles = [*simple_cycles_with_multiplicity(g)]
            curve_count = sum(m for _, m in cycles)
            if (
                curve_count in targets
                and (cc := curve_complex(cycles)) in targets[curve_count]
            ):
                result[indices[cc]].add(g)
            if curve_count < max_curve_count:
                for a, b in bridge_pairs(g):
                    descendants.add(bridge(g, a, b))

    return result


def edge_automorphism_group(g: DiGraph):
    lg: DiGraph = g.line_graph(immutable=False)
    lg.relabel(inplace=True)
    return lg.automorphism_group()


def subdivisions(g: DiGraph, n_verts):
    eaut = edge_automorphism_group(g)
    if eaut.is_trivial():
        return integer_vectors_inplace(n_verts, g.n_edges())
    else:
        return IntegerVectorsModPermutationGroup_with_constraints(
            eaut,
            n_verts,
            None,
            sgs=tuple((*s,) for s in eaut.strong_generating_system()),
        )


def is_orbit_rep(vec: list[int], orbits: Orbits) -> bool:
    for orbit in orbits:
        for i, j in pairwise(orbit):
            if vec[i] > vec[j]:
                return False
    else:
        return True


def subdivvy(g, n_verts, use_orbits=True):
    retz = []
    A = np.zeros((n_verts, n_verts), dtype=np.int_)
    edges = g.edges(labels=False)
    ekey = [(u, v, i) for i, (u, v) in enumerate(edges)]
    print(f"edges = {ekey}")
    DiGraph(ekey, format="list_of_edges", multiedges=True, loops=True).show(
        edge_labels=True
    )
    n = len(g)
    for subdivs in (
        subdivisions(g, n_verts - len(g))
        if use_orbits
        else integer_vectors_inplace(n_verts - len(g), g.n_edges())
    ):  # inplace_int_vectors(n_verts - len(g), g.n_edges()):
        A.fill(0)
        j = n
        for i, (v0, v1) in enumerate(edges):
            for _ in range(subdivs[i]):
                A[v0][j] += 1
                v0 = j
                j += 1
            A[v0][v1] += 1

        mat = matrix(A)
        print(f"{mat}")
        print(f"λ = {max(np.linalg.eigvals(A))}")
        ng = DiGraph(matrix(A), format="adjacency_matrix", multiedges=True, loops=True)
        # ng.show()
        retz.append(ng)

    return retz


def approx_stretch_factor(A, tol=0):
    return np.real_if_close(
        eigs(A, k=1, which="LR", return_eigenvectors=False, tol=tol)[0]
    )


def exact_stretch_factor(A):
    return max(matrix(A).eigenvalues())


def stretch_factor_comparison_function(max_lambda, tol=0):
    if max_lambda is None:
        return lambda A: True
    else:
        max_lambda_N = np.float64(max_lambda)
        if isinstance(max_lambda, AlgebraicNumber):
            return lambda A: (
                (sf := approx_stretch_factor(A, tol)) <= max_lambda_N
                or (
                    np.isclose(sf, max_lambda_N)
                    and exact_stretch_factor(A) <= max_lambda
                )
            )
        else:
            return lambda A: (
                (sf := approx_stretch_factor(A, tol)) <= max_lambda_N
                or np.isclose(sf, max_lambda_N)
            )


def candidate_digraphs(spec, max_lambda=None, tol=0):
    result = []
    is_admissible = stretch_factor_comparison_function(max_lambda, tol)
    ccs, nvs = zip(
        *(
            (
                named_curve_complexes[cc] if isinstance(cc, str) else cc,
                ns if isinstance(ns, list | tuple | set) else [ns],
            )
            for cc, ns in (spec.items() if hasattr(spec, "items") else spec)
        )
    )
    for cc, topo_types, nvs in zip(
        ccs, topological_digraphs_with_curve_complexes(ccs), nvs
    ):
        for n_verts in nvs:
            A = np.zeros((n_verts, n_verts), dtype=np.int_)
            for g in topo_types:
                edges = g.edges(labels=False)
                n = len(g)
                for subdivs in subdivisions(g, n_verts - n):
                    A.fill(0)
                    j = n
                    for i, (v0, v1) in enumerate(edges):
                        for _ in range(subdivs[i]):
                            A[v0][j] += 1
                            v0 = j
                            j += 1
                        A[v0][v1] += 1
                    if is_admissible(A):
                        dg = DiGraph(
                            matrix(A),
                            format="adjacency_matrix",
                            multiedges=True,
                            loops=True,
                        )
                        dg.relabel(lambda i: i + 1)
                        result.append(dg)
    return result
