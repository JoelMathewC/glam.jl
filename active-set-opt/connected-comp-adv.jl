using Finch

function connected_components_adv_einsum(adj_matrix, max_iter)
    (n, _) = size(adj_matrix)

    G_prev = Tensor(Dense(SparseList(Element(false))), adj_matrix)
    G = Tensor(Dense(SparseList(Element(false))), adj_matrix)
    

    C = Tensor(Dense(SparseList(Element(false))), n, n)
    @finch begin
        for i in 1:n
            C[i,i] = true
        end
    end
    @einsum C[i,j] = C[i,j] | G[i,j]

    active_prev = Tensor(SparseByteMap(Pattern()), n)
    @einsum active_prev[i] = active_prev[i] | true
    active = Tensor(SparseByteMap(Pattern()), n)
    any_active = Scalar(false)

    for t in 2:max_iter
        @einsum G[i,j] = G_prev[i,j]
        
        active = Tensor(SparseByteMap(Pattern()), n)
        @finch begin
            for j = _
                for i = _
                    if active_prev[i] || active_prev[j]
                        for k = _
                            let d = G_prev[i,k] * G_prev[k,j]
                                G[i,j] |= (d != 0)
                                C[i,j] |= (d != 0)

                                active[i] |= xor(d != 0,G_prev[i,j])
                                active[j] |= xor(d != 0,G_prev[i,j])
                            end
                        end
                    end
                end
            end
        end

        @finch begin
            any_active .= false
            for i = _
                any_active[] |= active[i]
            end
        end
        if !any_active[]
            break
        end

        G_prev, G = G, G_prev
        active_prev, active = active, active_prev
    end

    # Since graph is directed
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
    
result = connected_components_adv_einsum(adj_matrix,10)

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