# ECS Design Notes

## Why ECS

- Better cache locality than deep object graphs.
- Predictable iteration over homogeneous data.
- Flexible composition (components over inheritance trees).

## Storage Strategy

Current implementation uses sparse-set component stores:

- dense component vectors
- dense entity vectors
- sparse entity-to-dense index lookup

This gives O(1) insert/remove (swap-remove) and contiguous query iteration.

## Query and Change Detection

- `query<T>()` returns dense snapshots for system iteration.
- `par_query<T>()` enables rayon-parallel workloads.
- `changed<T>()` supports incremental update systems.

## Next Steps

- Add archetype groups for multi-component joins.
- Add borrow-checked system parameter API.
- Add schedule dependency solver and conflict analysis.
