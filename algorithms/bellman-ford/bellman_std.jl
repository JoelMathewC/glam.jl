# ----------------------------------------------------------------------
# ALGORITHM: STANDARD BELLMAN FORD
# ASSUMED INPUTS:
#    1. Adjacency Matrix (G)
#    2. Starting node index (s)
#
# COMMENTS
#   1. This is a case of an iterative algorithm that needs to loop over
#      edges in a graph.
# ----------------------------------------------------------------------

distances = fill(typemax(Int), G.num_nodes());
frontier = zeros(Int, G.num_nodes()); 
frontier[s] = 1;

Graph-Iterate(frontier)
    Process
        if t > G.size()
            throw("Negative cycles exist!")
        end
        
        mask = frontier{t}[i] + G[i,j] >= frontier{t}[j] ? 0 : 1
        distances[j] = min(frontier{t}[j], frontier{t}[i] + G[i,j])
        frontier{t+1}[j] = mask * distances[j]
        
    end
    
    Stop frontier[] = 0