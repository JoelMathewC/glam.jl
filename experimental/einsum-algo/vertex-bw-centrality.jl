using Finch

# Brandes algorithm
function vertex_betweeness_centrality(adj_matrix)
    G = Tensor(Dense(SparseList(Element(0))), adj_matrix)
    (n, _) = size(G)

    bc_score = Tensor(Dense(Element(0)), n)
    bc_update = Tensor(Dense(Element(0)), n)

    for v in 1:n
        frontier_stack = []
        num_short_path = Tensor(SparseByteMap(Element(0)), n)
        @finch num_short_path[v] = 1

        frontier = Tensor(SparseByteMap(Element(0)), n)
        @finch for k in 1:n
            frontier[k] = G[v,k]
        end

        d = 0
        @einsum fnz_count[] += frontier[i]
        while fnz_count[] != 0
            d = d + 1
            push!(frontier_stack, frontier)
            @einsum num_short_path[i] = num_short_path[i] + frontier[i]
            @einsum frontier[j] += frontier[i] * G[i,j] * (num_short_path[j] == 0)
            @einsum fnz_count[] += frontier[i]
        end

        bc_update = Tensor(Dense(Element(0)), n)
        while d > 1
            frontier = pop!(frontier_stack)
            @einsum w[i] = (num_short_path[i] != 0) * frontier[i] * (1+bc_update[i])/num_short_path[i]
            @einsum w[i] += G[i,j] * w[j]

            frontier_prime = frontier_stack[end]
            @einsum w[i] = w[i] * frontier_prime[i] * num_short_path[i]
            @einsum bc_update[i] = bc_update[i] + w[i]
            d = d-1
        end

        @einsum bc_score[i] = bc_score[i] + bc_update[i]
    end

    return bc_score
end

adj_matrix = [
    0 1 1 1 0 0 0 0
    0 0 1 0 1 0 0 0
    0 0 0 0 1 0 0 0
    0 0 1 0 1 0 0 0
    0 0 0 0 0 1 1 0
    0 0 0 0 0 0 0 1
    0 0 0 0 0 0 0 1
    0 0 0 0 0 0 0 0
]

result = vertex_betweeness_centrality(adj_matrix)

for i in 1:size(adj_matrix,1)
    print("$(result[i]) ")
end