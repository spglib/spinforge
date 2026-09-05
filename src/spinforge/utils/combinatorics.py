from __future__ import annotations


def combine_dimensions(dimensions: list[int], max_dim: int) -> dict[int, list[list[int]]]:
    combined: list[tuple[int, list[int]]] = []
    for i, di in enumerate(dimensions):
        current = combined.copy()
        for count in range(1, max_dim // di + 1):
            items = [i for _ in range(count)]
            current.append((di * count, items))
            for dj, other in combined:
                if dj + di * count <= max_dim:
                    current.append((dj + di * count, other + items))
        combined = current

    combined_with_dims: dict[int, list[list[int]]] = {dim: [] for dim in range(1, max_dim + 1)}
    for dim, items in combined:
        combined_with_dims[dim].append(items)
    return combined_with_dims
