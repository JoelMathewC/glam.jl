# ----------------------------------------------------------------------
# ALGORITHM: STANDARD FLOYD WARSHALL
# ASSUMED INPUTS:
#    1. Adjacency Matrix (G)
#    2. Starting node index (s)
#    3. Number of vertices (N)
#
# COMMENTS
#   1. This code introduces a new construct Process([j,k]) that is yet 
#      to be adopted into the specification.
# ----------------------------------------------------------------------

cost = fill(typemax(Int), (N,N));
frontier = zeros(Int, N); 
frontier[s] = 1;

Graph-Iterate{t}(frontier)
    Process(i : frontier)
        cost[{j,N},{k,N}] = min(cost[j,k], G[j,i] + G[i,k])
    Post-Process
        frontier{t+1}[i+1] = 1;
    end
    
    Stop t > N