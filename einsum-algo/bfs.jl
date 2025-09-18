using Finch

function bfs_einsum(adj_matrix,src)
    G = Tensor(Dense(SparseList(Element(false))), adj_matrix)
    (n, _) = size(G)

    level = Tensor(SparseByteMap(Element(0)), n)
    visited = Tensor(SparseByteMap(Element(false)), n)
    frontier = Tensor(SparseByteMap(Element(false)), n)
    @finch frontier[src] = true

    iter_count = 1
    fnz_count = Scalar(1)

    while fnz_count[] > 0
        @einsum level[i] = level[i] | (frontier[i] * iter_count)
        @einsum visited[i] = (visited[i] | frontier[i])
        @einsum frontier[j] <<choose(false)>>= frontier[i] * G[i,j] * !visited[j]

        @einsum fnz_count[] += frontier[i]
        iter_count += 1
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