import datetime
import json
import pathlib
import math
from itertools import pairwise, product, islice
from collections import deque
from typing import (
    Generator,
    Iterable,
    Sequence,
    NamedTuple,
    cast,
)
import gc

import numpy as np
from sage.rings.integer import Integer
from sage.combinat.integer_vectors_mod_permgroup import (
    IntegerVectorsModPermutationGroup_with_constraints,
)
from sage.graphs.digraph import DiGraph
from sage.graphs.graph import Graph
from sage.matrix.constructor import matrix
from sage.rings.qqbar import AlgebraicNumber
from scipy.sparse.linalg import eigs
from sets.disjoint_set import DisjointSet_of_hashables
from sage.groups.perm_gps.partn_ref.refinement_graphs import search_tree

from utils import (
    grouped_by,
    edge_multiplicities,
    first,
    rle,
    mdg_str,
)

type Edge = tuple[int, int]
type CycleWithMultiplicity = tuple[list[int], int]

type AutGens = Sequence[Sequence[int]]
type DegList = list[int]
type DeletionCandidates = EdgeCandidates | VertexCandidates


class EdgeCandidates(NamedTuple):
    items: set[Edge]
    du: int
    dv: int
    mult: int
    loop: bool


class VertexCandidates(NamedTuple):
    items: set[int]
    du: int
    dv: int


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


def simple_cycles_and_curve_count(g: DiGraph):
    get_edge_mult = edge_multiplicities(g).__getitem__
    cycles = [*simple_cycles(g)]
    return cycles, sum(
        math.prod(map(get_edge_mult, pairwise(cycle))) for cycle in cycles
    )


def simple_cycles_and_curve_count_2(g: DiGraph):
    cycles = [*simple_cycles_with_multiplicity(g)]

    return cycles, sum(m for _, m in cycles)


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
    name: Graph(spec, name=name).canonical_label(immutable=True)
    for name, spec in [
        # ("A1", 1),
        ("2A1", 2),
        ("3A1", 3),
        ("4A1", 4),
        ("5A1", 5),
        ("6A1", 6),
        ("7A1", 7),
        # ("A2", [[0, 1], [(0, 1)]]),
        ("A2*", [[0, 1, 2], [(0, 1)]]),
        ("A2**", [[0, 1, 2, 3], [(0, 1)]]),
        ("A2***", [[0, 1, 2, 3, 4], [(0, 1)]]),
        ("A3*", [[0, 1, 2, 3], [(0, 1), (1, 2)]]),
        ("Y*", [[0, 1, 2, 3, 4], [(0, 1), (0, 2), (0, 3)]]),
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


def write_topological_digraphs_with_curve_complexes(
    ccs_dict: dict[str, Graph],
    out_dir: str = "../data/topological_graphs/",
    log_every: int = 25_000,
    max_size_before_writing=10_000,
    n_expected=None,
):
    def log(*args):
        print(f"{datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S]:')}", *args)

    start_time = datetime.datetime.now()
    output_path = pathlib.Path(out_dir) / f"{start_time.strftime('%Y%m%d%H%M%S')}/"
    output_path.mkdir()

    names = {g.canonical_label(immutable=True): name for name, g in ccs_dict.items()}
    targets = grouped_by(names.keys(), len)
    results = {name: deque[DiGraph]() for name in ccs_dict}
    n_found = 0
    n_written = dict.fromkeys(ccs_dict, 0)
    n_examined = 0
    outfiles = dict.fromkeys(ccs_dict, None)

    max_curve_count = max(targets)

    def write_graphs(cc_name, graphs):
        out_fname = outfiles[cc_name]
        if out_fname is None:
            outfiles[cc_name] = out_fname = (
                output_path / f"{cc_name.replace('*', '_star')}.graphs"
            )

        n_written[cc_name] += len(graphs)
        with out_fname.open("a") as file:
            for g in graphs:
                file.write(
                    f"{json.dumps([*g.edge_iterator(labels=False)], separators=(',', ':'))}\n"
                )

        log(f"wrote {len(graphs)} graphs to {out_fname}")

    log(f"writing results to {output_path}")
    log(f"targets: {', '.join(ccs_dict.keys())}")
    log(f"max_curve_count: {max_curve_count}")

    def go(
        g: DiGraph,
        aut_gens: AutGens,
        cands: DeletionCandidates,
        idegs: DegList,
        odegs: DegList,
    ):
        nonlocal n_examined
        nonlocal n_found
        n_examined += 1
        if n_examined % log_every == 0:
            log(
                f"{n_found}{f' ({100 * n_found / n_expected:.2f}%)' if n_expected else ''} found:\n"
                f"\t{' '.join(f'{name}: {len(rs) + n_written[name]:<5}' for name, rs in results.items())}"
            )
            n_examined = 0

        n = len(g)
        n_v2vs = (
            0
            if idegs[n - 1] != 1 or odegs[n - 1] != 1
            else 1
            if n == 1 or idegs[n - 2] != 1 or odegs[n - 2] != 1
            else 2
        )

        if isinstance(cands, EdgeCandidates):  # previous augmentation was an edge
            cycles = [*simple_cycles_with_multiplicity(g)]
            curve_count = sum(m for _, m in cycles)
            if (
                n_v2vs == 0
                and curve_count in targets
                and (cc := curve_complex(cycles)) in targets[curve_count]
            ):
                coll = results[names[cc]]
                coll.append(g)
                n_found += 1

                if len(coll) >= max_size_before_writing:
                    write_graphs(names[cc], coll)
                    coll.clear()
                    gc.collect()

            if curve_count >= max_curve_count:
                return

        # union-find representatives of edge orbits under Aut(g)
        orbits = DisjointSet_of_hashables(product(range(n), range(n)))
        for gen in aut_gens:
            for u in range(n):
                for v in range(n):
                    orbits.union((u, v), (gen[u], gen[v]))

        # with 0 valence-two vertices:
        #   add an edge btwn any two verts
        #   subdivide an existing edge
        # with 1 valence-two vertex:
        #   add an edge btwn a valence-two vert and any other vert
        #   subdivide an existing edge
        # with 2 valence-two vertices:
        #   add an edge btwn two valence-two verts

        if n_v2vs == 0:
            to_add = (r for e, r in orbits.__getstate__() if e == r)
        elif n_v2vs == 1:
            to_add = (r for e, r in orbits.__getstate__() if e == r and n - 1 in r)
        else:
            to_add = {orbits.find((n - 1, n - 2)), orbits.find((n - 2, n - 1))}

        if n_v2vs < 2:
            to_subdiv = {
                orbits.find(e): e for e in g.edge_iterator(labels=False)
            }.values()
            if isinstance(cands, EdgeCandidates):

                def should_skip_addition(u: int, v: int):
                    if u != v:
                        if cands.loop:
                            return True
                        mult = len(g.edge_label(u, v)) + 1 if g.has_edge(u, v) else 1
                        if mult < cands.mult:
                            return True
                        elif mult == cands.mult:
                            du = odegs[u] + 1
                            if du < cands.du:
                                return True
                            elif du == cands.du:
                                if idegs[v] + 1 < cands.dv:
                                    return True
                    return False

                should_skip_subdiv = False
            else:
                assert isinstance(cands, VertexCandidates)
                should_skip_addition = False

                def should_skip_subdiv(u: int, v: int):
                    if odegs[u] < cands.du:
                        return True
                    return False

        else:
            to_subdiv = ()
            should_skip_addition = False
            should_skip_subdiv = False

        for u, v in to_add:
            if should_skip_addition and should_skip_addition(u, v):
                continue
            z = g.copy()
            z.add_edge(u, v)
            z_idegs = [*idegs]
            z_odegs = [*odegs]
            z_odegs[u] += 1
            z_idegs[v] += 1
            z_cands = canonical_deletion_candidates(z, z_idegs, z_odegs)
            if (
                isinstance(z_cands, EdgeCandidates)
                and z_cands.du == z_odegs[u]
                and z_cands.dv == z_idegs[v]
            ):
                z_aut_gens, clabel = aut_gens_and_can_label(z)
                max_e: Edge = max(
                    z_cands.items, key=lambda e: (clabel[e[0]], clabel[e[1]])
                )
                if same_edge_orbit(z_aut_gens, (u, v), max_e):
                    go(z, z_aut_gens, z_cands, z_idegs, z_odegs)

        for u, v in to_subdiv:
            if should_skip_subdiv and should_skip_subdiv(u, v):
                continue

            z = g.copy()
            w = subdivide_edge(z, u, v)
            z_idegs = [*idegs, 1]
            z_odegs = [*odegs, 1]
            z_cands = canonical_deletion_candidates(z, z_idegs, z_odegs)
            if (
                isinstance(z_cands, VertexCandidates)
                and z_cands.du == z_odegs[u]
                and z_cands.dv == z_idegs[v]
            ):
                z_aut_gens, clabel = aut_gens_and_can_label(z)
                max_v = max(z_cands.items, key=clabel.__getitem__)
                if same_vert_orbit(z_aut_gens, w, max_v):
                    go(z, z_aut_gens, z_cands, z_idegs, z_odegs)

    go(
        DiGraph([(0, 0)], format="list_of_edges", multiedges=True, loops=True),
        (),
        EdgeCandidates({(0, 0)}, 2, 2, 1, True),
        [1],
        [1],
    )

    while results:
        name, result = results.popitem()
        write_graphs(name, list(result))

    delta = datetime.datetime.now() - start_time
    log(
        f"finished in {f'{delta.days}d ' if delta.days > 0 else ''}{
            (datetime.datetime.min + delta).strftime('%-Hh %-Mm %-Ss')
        }"
    )


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


def approx_stretch_factor_array(A, tol=0):
    return np.real_if_close(
        eigs(A, k=1, which="LR", return_eigenvectors=False, tol=tol)[0]
    )


def exact_stretch_factor_array(A):
    return max(matrix(A).eigenvalues())


def exact_stretch_factor(data):
    if isinstance(data, DiGraph):
        return max(data.adjacency_matrix().eigenvalues())
    else:
        return max(matrix(data).eigenvalues())


def charpoly(g: DiGraph):
    return g.adjacency_matrix().characteristic_polynomial()


def stretch_factor_comparison_function(max_lambda, tol=0):
    if max_lambda is None:
        return lambda A: True
    else:
        max_lambda_N = np.float64(max_lambda)
        if isinstance(max_lambda, AlgebraicNumber):
            return lambda A: (
                (sf := approx_stretch_factor_array(A, tol)) <= max_lambda_N
                or (
                    np.isclose(sf, max_lambda_N)
                    and exact_stretch_factor_array(A) <= max_lambda
                )
            )
        else:
            return lambda A: (
                (sf := approx_stretch_factor_array(A, tol)) <= max_lambda_N
                or np.isclose(sf, max_lambda_N)
            )


def subdivide_edges(
    g: DiGraph, subdivs: Sequence[int] | dict[Edge, int | Iterable[int]], inplace=False
):
    """
    Subdivide the edges of ``g``.

    INPUT:

    - ``g`` -- a digraph
    - ``subdivs`` -- subdivisions, either:
        - a list of integers, where the `i`th entry is the number of times to subdivide ``g.edges()[i]``
        - a dictionary with entries of the form ``((u, v), n)``, indicating that the edge from ``u`` to ``v``
          should be subdivided ``n`` times.
          If ``g`` has multiple edges from ``u`` to ``v``, then ``n`` can be a list indicating how
          many times to subdivide each one. Otherwise, only the first edge from ``u`` to ``v`` is subdivided.
    - ``inplace`` -- boolean (default False) indicating whether to modify the original ``g``,
      or return a copy with edges subdivided.

    OUTPUT:

    A ``DiGraph`` representing ``g`` with subdivided edges.
    """
    if not inplace:
        g = g.copy()

    if isinstance(subdivs, dict):
        for e, ns in subdivs.items():
            if isinstance(ns, int | Integer):
                ns = (ns,)
            for n in ns:
                u, v = e
                for _ in range(n):
                    u = subdivide_edge(g, u, v)
    else:
        for i, (u, v) in enumerate(list(g.edge_iterator(labels=False))):
            for _ in range(subdivs[i]):
                u = subdivide_edge(g, u, v)

    return g


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


def roots(ds):
    return tuple(
        root
        for item, root in (
            ds.__getstate__()
            if isinstance(ds, DisjointSet_of_hashables)
            else enumerate(ds.__getstate__())
        )
        if item == root
    )


def aut_gens_and_can_label(g: DiGraph):
    from graphs.generic_graph import graph_isom_equivalent_non_edge_labeled_graph

    G, partition, relabeling = graph_isom_equivalent_non_edge_labeled_graph(
        g, partition=[[*range(len(g))]], return_relabeling=True
    )
    HB = DiGraph(len(G), loops=True)._backend
    for u, v in G.edge_iterator(labels=False):
        HB.add_edge(u, v, None, True)
    a, b, c = search_tree(HB.c_graph()[0], partition, certificate=True, dig=True)
    return a, {v: c[relabeling[v]] for v in g}


def is_strong_bridge(g: DiGraph, u: int, v: int):
    """
    Check whether the directed edge ``(u, v)`` is a strong bridge in the digraph ``g``.

    A strong bridge is an edge of ``g`` whose removal increases the number of strongly connected components.
    """
    if u == v or len(g.edge_label(u, v)) > 1:
        return False
    seen = {u}
    unchecked = deque(w for w in g.neighbors_out(u) if w != u and w != v)
    while unchecked:
        u = unchecked.popleft()
        for w in g.neighbors_out(u):
            if w == v:
                return False
            if w not in seen:
                seen.add(w)
                unchecked.append(w)
    return True


def is_cut_edge_old(g: DiGraph, u: int, v: int):
    gcp = g.copy()
    gcp.delete_edge(u, v)
    return not gcp.is_strongly_connected()


def canonical_deletion_candidates(
    g: DiGraph,
    idegs: DegList | None = None,
    odegs: DegList | None = None,
    verbose=False,
) -> DeletionCandidates:

    n = len(g)
    if idegs is None:
        idegs = list(g.in_degree(range(n)))
    if odegs is None:
        odegs = list(g.out_degree(range(n)))

    if verbose:
        print(f"canonical_deletion_candidates({mdg_str(g)}, {idegs}, {odegs})")

    if n == 1:
        if g.n_edges() == 0:
            return VertexCandidates({0}, 0, 0)
        else:
            return EdgeCandidates({(0, 0)}, odegs[0], idegs[0], g.n_edges(), True)

    du_max: int = 0
    dv_max: int = 0
    if v2vs := [i for i in range(n) if idegs[i] == 1 == odegs[i]]:
        if verbose:
            print(f"valence-2 vertices: {v2vs}")
        candidates: set[int] = set()
        for w in v2vs:
            du = cast(int, odegs[first(g.neighbors_in(w))])
            dv = cast(int, idegs[first(g.neighbors_out(w))])
            if verbose:
                print(
                    f"status: candidates = {candidates}; "
                    f"du_max = {du_max}, dv_max = {dv_max}"
                )
                print(f"w = {w}; du = {du}, dv = {dv}")
            if du >= du_max:
                if du > du_max:
                    du_max = du
                    dv_max = dv  # dv >= dv_max
                    candidates.clear()
                if dv >= dv_max:  # du == du_max
                    if dv > dv_max:
                        dv_max = dv
                        candidates.clear()
                    candidates.add(w)

        if verbose:
            print(
                f"final status: candidates = {candidates}; "
                f"du_max = {du_max}, dv_max = {dv_max}"
            )

        return VertexCandidates(candidates, du_max, dv_max)
    else:
        candidates = set[Edge]()
        mult_max = 1
        loop = False
        e: tuple[int, int]
        for e, mult in rle(g.edge_iterator(labels=False)):
            if verbose:
                print(
                    f"status: candidates = {candidates}; "
                    f"mult_max={mult_max}, du_max={du_max}, dv_max={dv_max}, loop={loop}"
                )
                print(f"e = {e}; mult = {mult}, du = {odegs[e[0]]}, dv = {idegs[e[1]]}")
            if e[0] == e[1]:
                if not loop:
                    loop = True
                    mult_max = mult
                    du_max = odegs[e[0]]
                    dv_max = idegs[e[1]]
                    candidates.clear()
                    candidates.add(e)
                    continue
            elif loop:
                continue

            if mult > mult_max:  # can't be a strong bridge
                mult_max = mult
                du_max = odegs[e[0]]
                dv_max = idegs[e[1]]
                candidates.clear()
                candidates.add(e)
            elif mult == mult_max and not is_strong_bridge(g, *e):
                du = odegs[e[0]]
                dv = idegs[e[1]]
                if du >= du_max:
                    if du > du_max:
                        du_max = du
                        dv_max = dv  # dv >= dv_max
                        candidates.clear()
                    if dv >= dv_max:  # du == du_max
                        if dv > dv_max:
                            dv_max = dv
                            candidates.clear()
                        candidates.add(e)

        if verbose:
            print(
                f"final status: candidates = {candidates}; "
                f"mult_max = {mult_max}, du_max = {du_max}, dv_max = {dv_max}, loop = {loop}"
            )
        return EdgeCandidates(candidates, du_max, dv_max, mult_max, loop)


def topological_digraphs_with_curve_complexes(
    ccs: Graph | Iterable[Graph], verbose=False
):
    if isinstance(ccs, Graph):
        single_input = True
        ccs = (ccs,)
    else:
        single_input = False
    indices = {g.canonical_label(immutable=True): i for i, g in enumerate(ccs)}
    targets = grouped_by(indices.keys(), len)
    results = tuple(deque() for _ in indices)
    max_curve_count = max(targets)

    def can_aug_sc(
        g: DiGraph,
        aut_gens: AutGens,
        prev_aug: DeletionCandidates,
        idegs: DegList,
        odegs: DegList,
    ):
        n = len(g)
        n_v2vs = (  # number of valence-two vertices
            0
            if idegs[n - 1] != 1 or odegs[n - 1] != 1
            else 1
            if n == 1 or idegs[n - 2] != 1 or odegs[n - 2] != 1
            else 2
        )
        success = False

        if verbose:
            print(f"can_aug_sc({mdg_str(g)}, {aut_gens}, {prev_aug}, {idegs}, {odegs})")
            g.show(vertex_color="lightgreen")

        # if the previous augmentation added an edge, recompute the curve complex
        if isinstance(prev_aug, EdgeCandidates):
            cycles = [*simple_cycles_with_multiplicity(g)]
            curve_count = sum(m for _, m in cycles)
            if (
                n_v2vs == 0
                and curve_count in targets
                and (cc := curve_complex(cycles)) in targets[curve_count]
            ):
                results[indices[cc]].append(g)
                success = True
                if verbose:
                    print("added!")
            # stop once the number of sccs exceeds the maximum allowable
            if curve_count >= max_curve_count:
                return success

        # union-find representatives of edge orbits under Aut(g)
        orbits = DisjointSet_of_hashables(product(range(n), range(n)))
        for gen in aut_gens:
            for u in range(n):
                for v in range(n):
                    orbits.union((u, v), (gen[u], gen[v]))

        # with 0 valence-two vertices:
        #   add an edge btwn any two verts
        #   subdivide an existing edge
        # with 1 valence-two vertex:
        #   add an edge btwn a valence-two vert and any other vert
        #   subdivide an existing edge
        # with 2 valence-two vertices:
        #   add an edge btwn two valence-two verts

        if n_v2vs == 0:
            to_add = (r for e, r in orbits.__getstate__() if e == r)
        elif n_v2vs == 1:
            to_add = (r for e, r in orbits.__getstate__() if e == r and n - 1 in r)
        else:
            to_add = {orbits.find((n - 1, n - 2)), orbits.find((n - 2, n - 1))}

        if n_v2vs < 2:
            to_subdiv = {
                orbits.find(e): e for e in g.edge_iterator(labels=False)
            }.values()
            if isinstance(prev_aug, EdgeCandidates):

                def should_skip_addition(u: int, v: int):
                    if u != v:
                        if prev_aug.loop:
                            return True
                        mult = len(g.edge_label(u, v)) + 1 if g.has_edge(u, v) else 1
                        if mult < prev_aug.mult:
                            return True
                        elif mult == prev_aug.mult:
                            du = odegs[u] + 1
                            if du < prev_aug.du:
                                return True
                            elif du == prev_aug.du:
                                if idegs[v] + 1 < prev_aug.dv:
                                    return True
                    return False

                should_skip_subdiv = False
            else:
                assert isinstance(prev_aug, VertexCandidates)
                should_skip_addition = False

                def should_skip_subdiv(u: int, v: int):
                    if odegs[u] < prev_aug.du:
                        return True
                    return False

        else:
            to_subdiv = ()
            should_skip_addition = False
            should_skip_subdiv = False

        for u, v in to_add:
            if should_skip_addition and should_skip_addition(u, v):
                continue
            z = g.copy()
            z.add_edge(u, v)
            z_idegs = [*idegs]
            z_odegs = [*odegs]
            z_odegs[u] += 1
            z_idegs[v] += 1
            z_cands = canonical_deletion_candidates(
                z, z_idegs, z_odegs, verbose=verbose
            )
            if verbose:
                print(f"augmenting {g.edges(labels=False)} with edge {u, v}")
            if (
                isinstance(z_cands, EdgeCandidates)
                and z_cands.du == z_odegs[u]
                and z_cands.dv == z_idegs[v]
            ):
                z_aut_gens, clabel = aut_gens_and_can_label(z)
                max_e: Edge = max(
                    z_cands.items, key=lambda e: (clabel[e[0]], clabel[e[1]])
                )
                if same_edge_orbit(z_aut_gens, (u, v), max_e):
                    success |= can_aug_sc(z, z_aut_gens, z_cands, z_idegs, z_odegs)
                elif verbose:
                    z.show(vertex_color="lightcoral")
                    print(f"e: {u, v}; max_e: {max_e}")
            elif verbose:
                z.show(vertex_color="lightcoral")
                print(f"e: {u, v}; cdc: {z_cands}")

        for u, v in to_subdiv:
            if should_skip_subdiv and should_skip_subdiv(u, v):
                continue

            z = g.copy()
            w = subdivide_edge(z, u, v)
            z_idegs = [*idegs, 1]
            z_odegs = [*odegs, 1]
            z_cands = canonical_deletion_candidates(
                z, z_idegs, z_odegs, verbose=verbose
            )
            if verbose:
                print(f"augmenting {g.edges(labels=False)} by subdividing {u, v}")
            if (
                isinstance(z_cands, VertexCandidates)
                and z_cands.du == z_odegs[u]
                and z_cands.dv == z_idegs[v]
            ):
                z_aut_gens, clabel = aut_gens_and_can_label(z)
                max_v = max(z_cands.items, key=clabel.__getitem__)
                if same_vert_orbit(z_aut_gens, w, max_v):
                    success |= can_aug_sc(z, z_aut_gens, z_cands, z_idegs, z_odegs)
                elif verbose:
                    z.show(vertex_color="lightcoral")
                    print(f"v: {w}; max_v: {max_v}")
            elif verbose:
                z.show(vertex_color="lightcoral")
                print(f"v: {w}; cdc: {z_cands}")

        return success

    can_aug_sc(
        DiGraph([(0, 0)], format="list_of_edges", multiedges=True, loops=True),
        (),
        EdgeCandidates({(0, 0)}, 2, 2, 1, True),
        [1],
        [1],
    )

    return results[0] if single_input else results


def relabeled_with_valence_two_vertices_last(g: DiGraph, inplace=False):
    if not inplace:
        g = g.copy()
    n = len(g)
    g.relabel()
    v2vs = g.vertices(degree=2)
    if v2vs:
        perm = [*range(n)]
        for i, v in enumerate(v2vs):
            perm[v], perm[n - 1 - i] = perm[n - 1 - i], perm[v]
        g.relabel(perm)

    return g


def canonical_deletion_path(g: DiGraph, show=True):
    if not g.is_strongly_connected():
        raise ValueError("graph is not strongly connected")
    if len(g.vertices(degree=2)) > 2:
        raise ValueError("graph has more than 2 valence-two vertices")

    g = relabeled_with_valence_two_vertices_last(g)
    paff = [g]
    while g.size():
        if show:
            g.show()
        g = relabeled_with_valence_two_vertices_last(g)
        cdcs = canonical_deletion_candidates(g)
        aut_gens, clabel = aut_gens_and_can_label(g)
        if isinstance(cdcs, VertexCandidates):
            can_del = max(cdcs.items, key=clabel.__getitem__)
            u = first(g.neighbors_in(can_del))
            v = first(g.neighbors_out(can_del))
            g.delete_vertex(can_del)
            g.add_edge(u, v)
        elif isinstance(cdcs, EdgeCandidates):
            can_del = max(cdcs.items, key=lambda e: (clabel[e[0]], clabel[e[1]]))
            g.delete_edge(can_del)
        if show:
            print(f"delete {can_del}")
        paff.append(g)

    if show:
        g.show()

    return paff


def same_edge_orbit(aut_gens: AutGens, e0: Edge, e1: Edge):
    """Determine whether the edge ``e1`` is contained in the orbit of ``e0`` under the group generated by ``aut_gens``."""
    if e0 == e1:
        return True
    elif not aut_gens or (e0[0] == e0[1]) != (e1[0] == e1[1]):
        return False
    seen = {e0}
    unchecked = deque(seen)
    while unchecked:
        e = unchecked.popleft()
        for gen in aut_gens:
            img_e = (gen[e[0]], gen[e[1]])
            if img_e == e1:
                return True
            if img_e not in seen:
                seen.add(img_e)
                unchecked.append(img_e)
    else:
        return False


def same_vert_orbit(aut_gens: AutGens, v0: int, v1: int):
    """Determine whether the vertex ``v1`` is contained in the orbit of ``v0`` under the group generated by ``aut_gens``."""
    if v0 == v1:
        return True
    elif not aut_gens:
        return False
    seen = {v0}
    unchecked = deque(seen)
    while unchecked:
        v = unchecked.popleft()
        for gen in aut_gens:
            img_v = gen[v]
            if img_v == v1:
                return True
            if img_v not in seen:
                seen.add(img_v)
                unchecked.append(img_v)
    else:
        return False


if __name__ == "__main__":
    tsg = DiGraph(
        [[0, 1, 2, 3], [(0, 1), (1, 2), (2, 3), (3, 0), (2, 1), (0, 3)]],
        multiedges=True,
        loops=True,
    )
    # print(canaug_traverse_strongly_connected(tsg, None, None, None))
