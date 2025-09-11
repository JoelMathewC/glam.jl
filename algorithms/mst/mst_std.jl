# ----------------------------------------------------------------------
# ALGORITHM: STANDARD KRUSKAL
# ASSUMED INPUTS:
#    1. Adjacency Matrix (G)
#    2. Index of the source node with the smallest index (s)
#    3. Number of vertices (N)
#
# COMMENTS
#   1. This is a case of an iterative algorithm that needs to loop over
#      edges in a graph.
#   2. This also depends on an accumulation operation after processing the
#      nodes in the frontier.
# ----------------------------------------------------------------------

cost = 0
frontier = zeros(Int, N); 
frontier[s] = 1;

Graph-Iterate{t}(frontier)
    Process(i : frontier)
        frontier{t+1}[{j,N}] = frontier{t}[i] + G[i,j]
        
    Post-Process
        (min_cost, min_cost_neighbor) = findmin(frontier{t+1}) 
        
        cost += min_cost
        MASK_NODE(min_cost_neighbor)
        
        frontier{t+1} .= 0
        frontier{t+1}[min_cost_neighbor] = 1
    end
    
    Stop frontier[] = 0