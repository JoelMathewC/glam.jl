# ----------------------------------------------------------------------
# ALGORITHM: STANDARD FLOYD WARSHALL
# ASSUMED INPUTS:
#    1. Adjacency Matrix (G)
#    2. Starting node index (s)
#
# COMMENTS
#   1. This code introduces a new construct Process([j,k]) that is yet 
#      to be adopted into the specification.
# ----------------------------------------------------------------------

cost = fill(typemax(Int), (G.num_nodes(),G.num_nodes()));
frontier = zeros(Int, G.num_nodes()); 
frontier[s] = 1;

Graph-Iterate(frontier)
    Process([j,k])
        cost[j,k] = min(cost[j,k], G[j,i] + G[i,k])
    Post-Process
        frontier{t+1}[i+1] = 1;
    end
    
    Stop t > G.num_nodes()