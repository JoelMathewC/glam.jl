# ----------------------------------------------------------------------
# ALGORITHM: BFS WITH PUSH-PULL OPTIMIZATIONS
# ASSUMED INPUTS:
#    1. Adjacency Matrix (G)
#    2. Starting node index (s)
#    3. Number of vertices (N)
#
# COMMENTS
# ----------------------------------------------------------------------

result = [];
frontier = zeros(Int, N); 
frontier[s] = 1;
do_push = true

Graph-Iterate{t}(frontier)
    Process(i : frontier)
        if do_push
            # Push
            frontier{t+1}[{j,N}] = frontier{t}[i] * G[i,j]
            result += i
            MASK_NODE(i)
        else
            # Pull
            frontier{t+1}[i] = ITER_TILL_FIRST_NON_ZERO{j,N}(frontier{t}[j] * G[j,i])

            if frontier{t+1}[i] > 0
                result += i
                MASK_NODE(i)
            end
        end
 
    Post-Process
        n_over_beta1 = N / 8.0
        n_over_beta2 = N / 512.0

        frontier_size_t = sum(frontier{t})
        frontier_size_t_1 = sum(frontier{t+1})

        if (frontier_size_t_1 > frontier_size_t) && (frontier_size_t_1 > n_over_beta1)
            do_push = false
        elseif (frontier_size_t > frontier_size_t_1) && (frontier_size_t_1 <= n_over_beta2)
            do_push = true
        end
    end
    
    Stop frontier[] = 0