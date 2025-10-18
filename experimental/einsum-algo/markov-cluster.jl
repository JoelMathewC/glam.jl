using Finch

function markov_clustering_einsum(adj_matrix, e, r, conv_thres)
    G = Tensor(Dense(SparseList(Element(0))), adj_matrix)
    (n, _) = size(G)

    # Total outgoing weight
    @einsum N_weight[j] += G[i,j]
    @einsum markov_mat[i,j] = (N_weight[j] != 0) * G[i,j] / N_weight[j]
    @einsum _markov_mat[i,j] = markov_mat[i,j]

    while true
        # expansion
        for _ in 1:(e-1)
            @einsum _markov_mat[i,j] += _markov_mat[i,k] * markov_mat[k,j]
        end

        # inflation
        @einsum _markov_mat[i,j] = _markov_mat[i,j] ^ r

        # normalization
        @einsum w[j] += _markov_mat[i,j]
        @einsum _markov_mat[i,j] = _markov_mat[i,j] / w[j]

        # convergence check
        @einsum magnitude[] += (_markov_mat[i,j] - markov_mat[i,j])^2
        if (magnitude[] ^ (1/2)) <= conv_thres
            return _markov_mat
        end

        @einsum markov_mat[i,j] = _markov_mat[i,j]
    end
end

adj_matrix = [   
        0 0 1 1 0 0 1 0
        0 0 1 1 0 0 0 1
        1 0 0 0 1 0 1 0
        1 1 0 0 0 1 0 0
        1 0 1 0 0 0 1 0
        0 0 0 1 0 0 1 1
        1 0 0 0 1 0 0 0
        0 1 0 1 0 1 0 0
    ]
    
result = markov_clustering_einsum(adj_matrix,2,2,1e-6)

for i in 1:size(adj_matrix,1)
    for j in 1:size(adj_matrix,1)
        if result[i,j] > 1e-4
            print("1 ")
        else
            print("0 ")
        end
    end
    print("\n")
end