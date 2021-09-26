import typing as tp

import numpy as np
import tqdm


def remove_back_edges(
    lil: tp.List[tp.List[int]], start: tp.Optional[tp.Iterable[int]] = None
) -> None:
    nn = len(lil)
    visited = np.zeros((nn,), dtype=bool)
    on_path = np.zeros((nn,), dtype=bool)
    num_edges = sum(len(l) for l in lil)
    prog = tqdm.tqdm(total=num_edges)

    if start is None:
        start = range(len(lil))

    # stack = list(start)

    # while stack:
    #     node = stack.pop()
    #     if visited[node]:
    #         continue
    #     assert not on_path[node]
    #     visited[node] = True
    #     on_path[node] = True
    #     neighbors = lil[node]
    #     prog.update(len(neighbors))
    #     for i in range(len(neighbors) - 1, -1, -1):
    #         neigh = neighbors[i]
    #         if on_path[neigh]:
    #             del neighbors[i]
    #     stack.extend(neighbors)
    #     on_path[node] = False

    def visit(node: int):
        if visited[node]:
            return
        assert not on_path[node]
        visited[node] = True
        on_path[node] = True
        neighbors = lil[node]
        for i in range(len(neighbors) - 1, -1, -1):
            prog.update()
            neigh = neighbors[i]
            if on_path[neigh]:
                # remove the path
                del neighbors[i]
            else:
                visit(neigh)

        on_path[node] = False

    for node in start:
        visit(node)
