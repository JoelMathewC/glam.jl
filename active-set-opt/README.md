# Active Set optimizations

1. Bellman Ford can be optimized this way
   1. The active set in this case is vertices whose distances to the source has updated in a single iteration
2. Floyd Warshall can be optimized this way
   1. The active set is list of vetrices with true/false values based on whether distances to any of their neighbours can been updated in the last iteration
3. Connected Components can be optimized this way
   1. The active set is list of vetrices with true/false values based on whether any one of the ith vertices had G[i,j] change.
4. Markov cluster expansion step
5. Triangle counting?

## Identifying applicability

We can identify that this optimization can be applied when doing codegen for an einsum by verifying that the einsum is of the following format
```
@einsum A[..] = A[..] + B[..] + ..
```
can this be A <red>= B[] + C[]?

Regarding codegen, note that A, B can be of any dimensions but to verify their inclusion in the active set we just need their first index
1. if `B[..]` is constant across iterations
   1. The active set is whatever is required to identify if A[i] has changed from the previous iteration
2. if `B[..]` is not constant across iterations
   1. The active is whatever is required to identify if either A[i], B[k] (and so on) has changed across iterations

The only problem at the moment is that it causes non-concordant traversal.