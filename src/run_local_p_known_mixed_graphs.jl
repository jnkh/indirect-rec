push!(LOAD_PATH, pwd())
using IndirectRec, GraphConnectivityTheory,GraphCreation
using PyCall, Distributions,JLD,LightGraphs


##facebook: N = 4040, k = 44
p = 0.1
num_trials = 1 #100
num_trials_perc = 200
graph_type_range = [:erdos_renyi, :watts_strogatz, :gamma_fb]#[:erdos_renyi,:watts_strogatz,:powerlaw_cluster,:fb]
graph_name_range = ["erdos_renyi", "watts_strogatz", "gamma_fb"]#["erdos renyi", "watts strogatz", "powerlaw cluster", "facebook"]
N_range = [100,200,500] 
C_range = [0.01,0.2,0.5,0.7] 
k_range = [5,10,30] 
n_range = [2,3,4,5,7,100]

#histogram data
hist_all_degrees = Array(Array{Any,1},(length(n_range),length(graph_type_range),length(N_range),length(C_range),length(k_range)))
hist_all_thresholds = Array(Array{Any,1},(length(n_range),length(graph_type_range),length(N_range),length(C_range),length(k_range)))
hist_all_clustering= Array(Array{Any,1},(length(n_range),length(graph_type_range),length(N_range),length(C_range),length(k_range)))
for (i,n) in enumerate(n_range)
    for (j,graph_type) in enumerate(graph_type_range)
        for (iN,N) in enumerate(N_range)
            for (iC,C) in enumerate(C_range)
                for (ik,k) in enumerate(k_range)
                num_trials_curr = graph_type == :fb ? 1 : num_trials
                hist_all_thresholds[i,j,iN,iC,ik] = Any[]
                hist_all_degrees[i,j,iN,iC,ik] = Any[]
                hist_all_clustering[i,j,iN,iC,ik] = Any[]
                for l = 1:num_trials_curr
                    g = create_graph(N,k,graph_type,C)
                    p_knowns = get_p_known_percolation(g,p,n,num_trials_perc)[1]
            #         p_knowns = p_knowns[p_knowns .> 0.0]
                    hist_all_degrees[i,j,iN,iC,ik] = vcat(hist_all_degrees[i,j,iN,iC,ik],LightGraphs.degree(g))
                    hist_all_thresholds[i,j,iN,iC,ik] = vcat(hist_all_thresholds[i,j,iN,iC,ik],p_knowns)
                    hist_all_clustering[i,j,iN,iC,ik] = vcat(hist_all_clustering[i,j,iN,iC,ik],LightGraphs.local_clustering_coefficient(g))
                end
                end
            end
        end
        println("n = $n, graph type: $(graph_name_range[j])")

    end
end

JLD.save("../data/per_node_data/misc_graphs_$(now()).jld","hist_all_degrees",hist_all_degrees,
"hist_all_thresholds",hist_all_thresholds,"hist_all_clustering",hist_all_clustering,
"graph_type_range",graph_type_range,"graph_name_range",graph_name_range,"N",N,"k",k,
"n_range",n_range,"p",p,N_range,C_range,k_range)
