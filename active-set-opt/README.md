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