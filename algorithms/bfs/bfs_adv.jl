# ----------------------------------------------------------------------
# ALGORITHM: BFS WITH PUSH-PULL OPTIMIZATIONS
# ASSUMED INPUTS:
#    1. Adjacency Matrix (G)
#    2. Starting node index (s)
#
# COMMENTS
# ----------------------------------------------------------------------

result = [];
frontier = zeros(Int, G.num_nodes()); 
frontier[s] = 1;
do_push = true

Graph-Iterate(frontier)
    Process()
        if do_push
            # Push
            frontier{t+1}[j] = frontier{t}[i] * G[i,j]
        else
            # Pull
            # TODO: We need to add early exit
            frontier{t+1}[i] = frontier{t}[j] * G[j,i]
        end

        result += i
        MASK_NODE(i)
    Post-Process
        n_over_beta1 = G.num_nodes() / 8.0;
        n_over_beta2 = G.num_nodes() / 512.0;

        frontier_size_t = sum(frontier{t})
        frontier_size_t_1 = sum(frontier{t+1})

        if (frontier_size_t_1 > frontier_size_t) && (frontier_size_t_1 > n_over_beta1)
            do_push = false
        elseif (frontier_size_t > frontier_size_t_1) && (frontier_size_t_1 <= n_over_beta2)
            do_push = true
        end
    end
    
    Stop frontier[] = 0