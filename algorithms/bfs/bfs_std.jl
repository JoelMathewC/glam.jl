# ----------------------------------------------------------------------
# ALGORITHM: STANDARD BFS
# ASSUMED INPUTS:
#    1. Adjacency Matrix (G)
#    2. Starting node index (s)
#
# COMMENTS
#   1. This is a case of an iterative algorithm that needs to loop over
#      nodes in a graph.
# ----------------------------------------------------------------------

result = [];
frontier = zeros(Int, G.num_nodes()); 
frontier[s] = 1;

Graph-Iterate(frontier)
    Process
        frontier{t+1}[j] = (frontier{t}[i] * G[i,j])
        result += i
        MASK_NODE(i)
    end
    
    Stop frontier[] = 0