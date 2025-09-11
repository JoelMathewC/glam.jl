# ----------------------------------------------------------------------
# ALGORITHM: STANDARD BELLMAN FORD
# ASSUMED INPUTS:
#    1. Adjacency Matrix (G)
#    2. Starting node index (s)
#    3. Number of vertices (N)
#
# COMMENTS
#   1. This is a case of an iterative algorithm that needs to loop over
#      edges in a graph.
# ----------------------------------------------------------------------

distances = fill(typemax(Int), N);
frontier = zeros(Int, N); 
frontier[s] = 1;

Graph-Iterate{t}(frontier)
    Process(i : frontier)
        if t > G.size()
            throw("Negative cycles exist!")
        end
        
        mask[{j,N}] = frontier{t}[i] + G[i,j] >= frontier{t}[j] ? 0 : 1
        distances[{j,N}] = min(frontier{t}[j], frontier{t}[i] + G[i,j])
        frontier{t+1}[{j,N}] = mask[{j}] * distances[j]
        
    end
    
    Stop frontier[] = 0