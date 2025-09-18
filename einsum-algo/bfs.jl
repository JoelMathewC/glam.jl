using Finch

function bfs_einsum(adj_matrix,src)
    G = Tensor(Dense(SparseList(Element(0))), adj_matrix)
    (n, _) = size(G)

    # Initialization for level
    level = Tensor(Dense(SparseList(Element(0))), 1, n)

    # Initializations to calculate pending of each node
    pending = Tensor(Dense(SparseList(Element(1))), 1, n)

    # Declare the starting state of the frontier
    frontier = Tensor(Dense(SparseList(Element(0))), 1, n)
    @finch frontier[1,src] = 1

    iter_count = 0
    fnz_count = Scalar(1)

    # Loop till the frontier is no longer empty
    while fnz_count[] > 0
        iter_count += 1

        # Mark all nodes in current frontier with a given level
        @einsum level_prime[k,i] = (frontier[k,i] * iter_count) + level[k,i]
        (level, level_prime) = (level_prime, level)

        # Mark all nodes in the current frontier as not pending
        # Note: the complicated math here is the equivalent of doing <<choose(1)>>
        # I'm avoiding using choose since it isnt working at the moment.
        @einsum pending_prime[i,j] = pending[i,j] - (frontier[i,j] * pending[i,j])
        (pending, pending_prime) = (pending_prime, pending)

        # Calculate the next frontier
        @einsum frontier_prime[k,j] = frontier[k,i] * (G[i,j] * pending[k,j])
        (frontier, frontier_prime) = (frontier_prime, frontier)

        @einsum fnz_count[] += frontier[i,j]
    end

    return level
end


adj_matrix = [   
        0 1 1; 
        0 0 0; 
        0 0 0
    ]

# adj_matrix = [   
#         0 1 1 0 0 0; 
#         0 0 1 1 0 0; 
#         0 0 0 0 1 0;
#         0 0 0 0 0 0;
#         0 0 0 0 0 1;
#         0 0 0 1 0 0;
#     ]


print(bfs_einsum(adj_matrix,1))


# FOR DEBUGGING (drop this into the loop)
# print("Step $(iter_count)\n")
# print("Frontier: $frontier\n")
# print("Level: $level\n")
# print("Pending: $pending\n")
# print("-----------------------\n")