using Finch

function bellman_ford_einsum(adj_matrix,src)
    G = Tensor(Dense(SparseList(Element(10^8))), adj_matrix)
    (n, _) = size(G)

    D = Tensor(Dense(SparseList(Element(10^8))), 1, n)
    @finch D[1,src] = 0

    fnz_count = Scalar(10^8 * (n-1))

    for t in 1:n
        @einsum D[k,i] <<min>>= D[k,j] + G[j,i]
        @einsum fnz_count_prime[] += D[i,j]

        if t == n && (fnz_count_prime[] < fnz_count[])
            throw("Negative cycle detected!")
        end

        fnz_count = fnz_count_prime
    end

    return D
end

adj_matrix = [   
        0    1    1; 
        10^8 0    10^8; 
        10^8 10^8 0
    ]

print(bellman_ford_einsum(adj_matrix,1))

# Negative Cycle case
# adj_matrix = [   
#         0    10^8  1; 
#         1    0     10^8; 
#         10^8 -3    0
#     ]

# adj_matrix = [   
#         0    1    5    10^8 10^8 10^8; 
#         10^8 0    3    12   10^8 10^8; 
#         10^8 10^8 0    10^8 2    10^8;
#         10^8 10^8 10^8 0    10^8 10^8;
#         10^8 10^8 10^8 10^8 0    2;
#         10^8 10^8 10^8 2    10^8 0;
#     ]