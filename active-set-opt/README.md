# Active Set optimizations

1. Bellman Ford can be optimized this way
   1. The active set in this case is vertices whose distances to the source has updated in a single iteration
2. Floyd Warshall can be optimized this way
   1. The active set is list of vetrices with true/false values based on whether distances to any of their neighbours can been updated in the last iteration
3. Connected Components can be optimized this way
   1. The active set is list of vetrices with true/false values based on whether any one of the ith vertices had G[i,j] change.
4. Triangle counting?

## Identifying applicability

We can identify that this optimization can be applied when doing codegen for an einsum by verifying that the einsum is one of the following format
```
(1) @einsum A[i] <<idempotent-op>>= B[i] <<op>> C[i,j]
(2) @einsum A[i,j] <<idempotent-op>>= B[i,k] <<op>> C[k,j]
```
