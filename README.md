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
// The t token is used throughout the block uniquely identify a single iteration.
Graph-Iterate{t}(frontier)

    // SUBBLOCK #1: Single iteration definition
    // This subblock is executed repeatedly till the stopping condition is met.
    // The subblock contains two sections within it, namely Process and Post-Process
    // A single execution of this subblock executes Process for i = 1..NUM_NODES
    // and then executes the contents of Post-Process.
    //
    // NOTES
    // 1. The Process block allows us to use an index (`i` in this case) to reference the current vertex
    //    being processed.
    // 2. The Post-Process section can be ignored if it doesn't serve a purpose in the algorithm.
    // 3. During the execution of this subblock, we may occasionally need to reference a secondary index
    //    to give us nested loop like functionality. This achieved through the iterator definition syntax
    //    i.e. ..[{var1, LIMIT}] = ...var1... The variable token within the curly braces of the LHS define 
    //    the secondary iterator. It can take on values from 1..LIMIT  
    // 4. At each iteration, the frontier for the next iteration (i.e. frontier{t+1}) is initialized 
    //    as an array of zeros. Its expected that the processing of frontier{t} populates frontier{t+1}.
    Process(i : frontier)
        frontier{t+1}[{j,N}] = frontier{t}[i] * G[i,j]
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

