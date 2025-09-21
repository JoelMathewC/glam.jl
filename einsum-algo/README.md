# Inferences from writing graph algos in Einsum

1. Language Specification
    - Implement a few more graph algorithms to finalize on the spec for the language
    - The spec should clarify the grammar of the language
    - GLAM should compile down to use finch code (and not einsum)
    - The GLAM implementation might benefit from having pre-allocated output buffers instead of having that generated (as is the case with einsum)
      - This would be useful because `@einsum d[i] <<min>>= k[i]` will not actually do the min of `d[i]` and `k[i]` since the output vector is re-allocated initialized to `initial_value`
2. Optimization possibilities
    - Read more about active set optimizations and try to find algorithms (floyd warshall might) that benefit from it
    - Look through existing literature to identify other optimizations that are shared across graph algorithms. How do we identify that a certain optimization is applicable?
    - Should GLAM support advanced datastructures like queues to implement certain algorithms like prims more easily?
