using Finch

function connected_components_einsum(adj_matrix, max_iter)
    G = Tensor(Dense(SparseList(Element(false))), adj_matrix)
    (n, _) = size(G)

    C = Tensor(Dense(SparseList(Element(false))), n, n)
    @finch begin
        for i in 1:n
            C[i,i] = true
        end
    end

    for t in 1:max_iter
        if t > 1
            @einsum G[i,j] |= (G[i,k] != 0) * (G[k,j] != 0)
        end
        @einsum C[i,j] = C[i,j] | G[i,j]
    end

    @einsum result[i,j] = C[i,j] & C[j,i]
    return result
end

adj_matrix = [   
    0 1 0 0 0 0 0 0
    0 0 1 0 1 1 0 0
    0 0 0 1 0 0 1 0
    0 0 1 0 0 0 0 1
    1 0 0 0 0 1 0 0
    0 0 0 0 0 0 1 0
    0 0 0 0 0 1 0 1
    0 0 0 0 0 0 0 1
]
    
result = connected_components_einsum(adj_matrix,10)

for i in 1:size(adj_matrix,1)
    for j in 1:size(adj_matrix,1)
        if result[i,j]
            print("1 ")
        else
            print("0 ")
        end
    end
    print("\n")
end