using Finch

function connected_components_adv_einsum(adj_matrix, max_iter)
    (n, _) = size(adj_matrix)

    # We need two lists here since the computation of G at time t depends on D at time t-1
    G_prev = Tensor(Dense(SparseList(Element(false))), adj_matrix)
    G = Tensor(Dense(SparseList(Element(false))), adj_matrix)
    
    # Connected Component result matrix
    C = Tensor(Dense(SparseList(Element(false))), n, n)
    @finch begin
        for i in 1:n
            C[i,i] = true
        end
    end
    @einsum C[i,j] = C[i,j] | G[i,j]

    # We need two lists here since the computation of active at time t+1 depends on active at time t
    active = Tensor(SparseByteMap(Pattern()), n)
    active_next = Tensor(SparseByteMap(Pattern()), n)
    any_active = Scalar(false)

    @finch for i = _ 
        active[i] = true
    end

    for t in 2:max_iter
        # TODO: This will take O(N^2), improve using active_next set
        @einsum G[i,j] = G_prev[i,j]
        
        @finch begin
            active_next .= 0
            for j = _
                for k = _
                    if active[j] || active[k]
                        for i = _
                            let d = G_prev[i,k] * G_prev[k,j]
                                G[i,j] |= (d != 0)
                                C[i,j] |= (d != 0)

                                active_next[j] |= xor(d != 0,G_prev[i,j])
                            end
                        end
                    end
                end
            end
        end

        @finch begin
            any_active .= false
            for i = _
                any_active[] |= active_next[i]
            end
        end
        if !any_active[]
            break
        end

        G_prev, G = G, G_prev
        active, active_next = active_next, active
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