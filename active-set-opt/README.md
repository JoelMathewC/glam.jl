# Active Set optimizations

1. Bellman Ford can be optimized this way
   1. The active set in this case is vertices whose distances to the source has updated in a single iteration
2. Floyd Warshall can be optimized this way
   1. The active set is list of vetrices with true/false values based on whether distances to any of their neighbours can been updated in the last iteration
3. Connected Components can be optimized this way
   1. The active set is list of vetrices with true/false values based on whether any one of the ith vertices had G[i,j] change.
4. Markov cluster expansion step
5. page rank
   1. This is very similar to bellman ford, the transition matrix is constant but the page rank is constantly changing

## Identifying applicability

We can identify that this optimization can be applied when doing codegen for an einsum by verifying that the einsum is of the following format
```
@einsum A[..] = A[..] + B[..] + ..
```

Regarding codegen, not that A, B can be of any dimensions but to verify their inclusion in the active set we just need their first index
1. if `B[..]` is constant across iterations
   1. The active set is whatever is required to identify if A[i] has changed from the previous iteration
2. if `B[..]` is not constant across iterations
   1. The active is whatever is required to identify if either A[i], B[k] (and so on) has changed across iterations