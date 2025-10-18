using Finch

function page_rank_einsum(adj_matrix, d, max_iter, conv_thres)
    G = Tensor(Dense(SparseList(Element(0))), adj_matrix)
    (n, _) = size(G)

    # Total outgoing weight
    @einsum N_weight[i] += G[i,j]
    @einsum transition_mat[i,j] = G[i,j] / N_weight[i]

    # number of outgoing edges
    @einsum N[i] += (G[i,j] != 0)
    @einsum pr[i] = (N[i] != 0) * 1/N[i]

    for t in 1:max_iter
        @einsum update[i] += transition_mat[i,j] * pr[j]
        @einsum pr_new[i] =  (N[i] != 0) * ((1-d/N[i]) + (d * update[i]))
        @einsum magnitude[] += (pr_new[i] - pr[i])^2
        
        if (magnitude[] ^ (1/2)) <= conv_thres
            return pr_new
        end

        pr = pr_new
        print(pr)
    end

    return pr
end

adj_matrix = [   
        0    1    1; 
        0    0    0; 
        0    1    0
    ]
    
print(page_rank_einsum(adj_matrix,0.85,1000,1e-6))