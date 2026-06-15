import itertools
from collections import ChainMap
from typing import Container, cast, Iterable
from copy import copy
from itertools import starmap

from sage.graphs.digraph import DiGraph
from sage.sets.disjoint_set import DisjointSet_of_hashables as DisjointSet
from utils import sgn, first, inverse_path, turn, first_truthy, stack_graph

type Turn = tuple[int, int]
type Stacks = tuple[tuple[int, ...], ...]


class PartialGraphMap(ChainMap[int, int]):
    """ """

    edge_maps: dict[int, tuple[int, ...]]
    base_edges: tuple[int, ...]

    def __init__(self, *maps, edge_maps=None, base_edges: tuple = ()):
        self.edge_maps = edge_maps or {}
        self.base_edges = base_edges
        super().__init__(*maps)

    @classmethod
    def from_stacks(cls, stacks: Iterable[Iterable[int]]):
        base_dict = {}
        base_edges = []
        for stack in iter(stacks):
            stack = iter(stack)
            try:
                u = next(stack)
            except StopIteration:
                continue
            base_edges.append(u)
            for v in stack:
                base_dict[u] = v
                base_dict[-u] = -v
                u = v
        return cls(base_dict, base_edges=tuple(base_edges))

    @property
    def final_edges(self):
        return tuple(self._final_edge(e) for e in self.base_edges)

    @property
    def stacks(self):
        return tuple(tuple(self._stack_tail(e)) for e in self.base_edges)

    def is_complete(self):
        return all(e in self.edge_maps for e in self.final_edges)

    def _stack_tail(self, e: int):
        while e is not None:
            yield e
            if abs(e) in self.edge_maps:
                break
            e = self[e]

    def _final_edge(self, e: int):
        while True:
            if abs(e) in self.edge_maps or e not in self:
                return e
            e = self[e]

    def __str__(self):
        return f"{self.__class__.__name__}({', '.join(repr(m) for m in self.maps)}, base_edges={self.base_edges!r}, edge_maps={self.edge_maps!r})"

    def __repr__(self):
        return f"{self.__class__.__name__}({
            ', '.join(
                itertools.chain(
                    (
                        f'{e}->{im[0] if len(im) == 1 else im}'
                        for e, im in self.edge_maps.items()
                    ),
                    (
                        '->'.join(str(e) for e in stack)
                        for stack in self.stacks
                        if len(stack) > 1
                    ),
                )
            )
        })"

    def __missing__(self, key):
        return None

    def __call__(self, x):
        match x:
            case [*_]:
                return tuple(starmap(self.edge_image, x))
            case _:
                return self[x]

    def __eq__(self, other):
        return (
            isinstance(other, PartialGraphMap)
            and self.stacks == other.stacks
            and self.edge_maps == other.edge_maps
        )

    def __hash__(self):
        return hash((self.__class__, self.stacks, *self.edge_maps.items()))

    def vertex_image(self, v):
        return self[v]

    def edge_image(self, e):
        if e in self.edge_maps:
            return self.edge_maps[e]
        elif -e in self.edge_maps:
            return inverse_path(self.edge_maps[-e])
        else:
            return (self[e],)

    def turn_image(self, u, v) -> Turn | tuple[None, None | int]:
        return turn(self[u], self[v])

    def iterate_image(self, v):
        yield v
        while (v := self(v)) is not None:
            yield v

    def preimages(self, v):
        if isinstance(v, Container):
            return set(u for map in self.maps for u, fu in map.items() if fu in v)
        else:
            return set(u for map in self.maps for u, fu in map.items() if fu == v)

    # @staticmethod

    def turns(self):
        return list(
            (-a, b) if -a <= b else (b, -a)
            for path in self.edge_maps.values()
            if len(path) > 1
            for a, b in itertools.pairwise(path)
        )

    def is_illegal_turn(self, u, v):
        return self.any_illegal(((u, v),))

    def any_illegal(self, turns):
        seen = set[Turn]()
        for turn in turns:
            u, v = turn(*turn)
            while not (u is None or (u, v) in seen):
                if u == v:
                    return True
                seen.add((u, v))
                u, v = self.turn_image(u, v)
        else:
            return False

    def new_child(self, m=None, edge_maps=None):
        return self.__class__(
            m or {},
            *self.maps,
            edge_maps=self.edge_maps | (edge_maps or {}),
            base_edges=self.base_edges,
        )

    def new_edge_mapping(self, e, image):
        if e < 0:
            e = -e
            image = inverse_path(image)
        return self.new_child({e: image[0], -e: -image[-1]}, edge_maps={e: image})

    def _add_edge_mapping(self, e, image):
        if e < 0:
            e = -e
            image = inverse_path(image)
        self[e] = image[0]
        self[-e] = -image[-1]
        self.edge_maps[e] = image

    def is_train_track(self):
        return not self.any_illegal(self.turns())

    def identify_vertices(
        self, ds: DisjointSet, /, assert_train_track=False, inplace=False, min_subsets=1
    ) -> DisjointSet:
        if not inplace:
            ds = copy(ds)
        seen = set[Turn]()

        for turn in self.turns():
            u, v = turn
            while ds.number_of_subsets() > min_subsets:
                if u is None or (u, v) in seen:
                    break
                elif u == v and assert_train_track:
                    raise AssertionError(f"Turn {turn!r} is illegal", turn)
                else:
                    seen.add((u, v))
                    ds.union(u, v)
                    u, v = self.turn_image(u, v)
            else:
                return ds

        return ds

    def domain_partition(self):
        return self.identify_vertices(
            DisjointSet([k for stack in self.stacks for v in stack for k in (v, -v)]),
            inplace=True,
        )

    def _flip_stacks(self, idxs):
        relab = {}
        for i, stack in enumerate(self.stacks):
            s = -1 if i in idxs else 1
            for e in stack:
                relab[e] = s * e
                relab[-e] = s * -e

        result = PartialGraphMap.from_stacks(self.stacks)
        for e, img in self.edge_maps.items():
            result._add_edge_mapping(
                e,
                tuple(relab[img_e] for img_e in img)
                if relab[e] == e
                else tuple(-relab[img_e] for img_e in reversed(img)),
            )

        return result

    def canonical_representative(self, relabel_edges=True, certificate=False):
        stacks = [tuple(self._stack_tail(e)) for e in self.base_edges]
        images = []
        stack_index: dict[int, int] = {}
        n_stacks = len(stacks)

        for i, stack in enumerate(stacks):
            images.append((stack[-1], self.edge_image(stack[-1])))
            for e in stack:
                stack_index[e] = i

        graph = stack_graph(stacks, images, stack_index, canonicalize=True)

        A = first_truthy(
            (
                tuple(abs(img_e) for img_e in img)
                for e, img in images
                if len(img) > 1
                and stack_index[abs(img[0])] != stack_index[abs(img[-1])]
            ),
            (),
        )
        mults = {
            0: 1
            if first_truthy(
                stack_index[A[i]] - stack_index[A[-i - 1]] for i in range(len(A))
            )
            <= 0
            else -1
        }

        u: int
        for u, v in graph.depth_first_search(0, edges=True):
            mult_u = mults[u]
            for img_e in self.edge_image(stacks[u][-1]):
                if img_e is not None and stack_index[abs(img_e)] == v:
                    mults[v] = mult_u * sgn(img_e)
                    break

        relab = {}
        j = 1
        for i in range(n_stacks):
            mult_i = mults[i]
            for e in stacks[i]:
                new_e = mult_i * (j if relabel_edges else e)
                relab[e] = new_e
                relab[-e] = -new_e
                j += 1

        result = PartialGraphMap.from_stacks(
            tuple(tuple(abs(relab[e]) for e in stacks[i]) for i in range(n_stacks))
        )
        
        for i in range(n_stacks):
            e = stacks[i][-1]
            result._add_edge_mapping(
                relab[e], tuple(relab[img_e] for img_e in self.edge_image(e))
            )

        return result, {e: new_e for e, new_e in relab.items() if e > 0} if certificate else result
