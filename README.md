# GLAM.jl

GLAM is a compiler intended for iterative graph algorithms. It provides a clear language for defining graph operations while compiling to efficient code internally.

## Specification
```
Graph-Iterate(frontier)

    Process
        /* ... */
    Post-Process
        /* ... */
    end
    
    Stop /* Stopping Condition */
```

## Language Design
A more detailed description of the specification is as follows
```
// Code block that is to be compiled by GLAM. The frontier token is used 
// to identify the nodes in graph that need to be processed in a single iteration
Graph-Iterate(frontier)

    // SUBBLOCK #1: Single iteration definition
    // This subblock is executed repeatedly till the stopping condition is met.
    // The subblock contains two sections within it, namely Process and Post-Process
    // A single execution of this subblock executes Process for all non-zero values 
    // in the frontier and then executes the contents of Post-Process.
    //
    // Few variables have special meaning in the context of this subblock:
    //      i -> the index of the current node for which the block is being processed
    //      j -> any value from 1..NUM_NODES, used to index the destination node for an i - j edge
    //      t -> the index of the current iteration, used to index the current and next frontier
    //
    // NOTES
    // 1. i and j can only be used in the Process section and not the Post-Process section.
    // 2. The Post-Process section can be ignored if it doesn't serve a purpose in the algorithm.
    // 3. During the execution for a single frontier element, code referencing j, is 
    //    processed for all values of j from 1..NUM_NODES.
    // 4. At each iteration, the frontier for the next iteration (i.e. frontier{t+1}) is initialized 
    //    as an array of zeros. Its expected that the processing of frontier{t} populates frontier{t+1}.
    Process
        frontier{t+1}[j] = frontier{t}[i] * G[i,j]
    Post-Process
        cost += AGG(frontier{t+1})
    end
    
    // SUBBLOCK #2: Stopping condition
    // The frontier or iteration counter (t) can be referenced here to define the stopping condition.
    Stop frontier[] = 0;
```

### Helper functions for the Process/Post-Process sections
| Function Name | Purpose |
|-|-|
| `MASK_NODE(node_idx)` | Indicates that a node should not be processed in future iterations if it exists in the frontier |

