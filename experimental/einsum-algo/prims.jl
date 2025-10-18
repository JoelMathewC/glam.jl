using Finch

function prims_einsum(adj_matrix)
    mst_weight = 0

    G = Tensor(Dense(SparseList(Element(10^8))), adj_matrix)
    (n, _) = size(G)

    min_edge = Scalar(Inf => (0=>0))
    @finch for j in 1:n
        for i in 1:n
            min_edge[] <<minby>>= G[i,j] => (i => j)
        end
    end

    src = min_edge[][2][1]

    edge_set = Tensor(SparseByteMap(Element(0)), n)
    @finch edge_set[src] = 10^8

    d = Tensor(SparseByteMap(Element(10^8)), G[src,:])

    for t in 1:(n-1)
        min_edge = Scalar(Inf => 0)
        @finch for i in 1:n
            min_edge[] <<minby>>= (d[i] + edge_set[i]) => i
        end

        mst_weight += min_edge[][1]
        dst = min_edge[][2]
        @finch edge_set[dst] = 10^8

        d = Queue()
        @finch for i in 1:n
            d[i] <<min_queue>>= G[dst,i]
        end
    end

    return mst_weight
end

adj_matrix = [   
        0       1       1; 
        10^8    0    10^8; 
        10^8    1       0
    ]

print(prims_einsum(adj_matrix))