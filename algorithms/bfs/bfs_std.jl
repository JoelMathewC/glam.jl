# ----------------------------------------------------------------------
# ALGORITHM: STANDARD BFS
# ASSUMED INPUTS:
#    1. Adjacency Matrix (G)
#    2. Starting node index (s)
#    3. Number of vertices (N)
#
# COMMENTS
#   1. This is a case of an iterative algorithm that needs to loop over
#      nodes in a graph.
# ----------------------------------------------------------------------

result = [];
frontier = zeros(Int, N); 
frontier[s] = 1;

Graph-Iterate{t}(frontier)
    Process(i : frontier)
        frontier{t+1}[{j,N}] = (frontier{t}[i] * G[i,j])
        result += i
        MASK_NODE(i)
    end
    
    Stop frontier[] = 0