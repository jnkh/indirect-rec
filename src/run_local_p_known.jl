push!(LOAD_PATH, pwd())
using IndirectRec, GraphConnectivityTheory,GraphCreation
using PyCall, Distributions,JLD,LightGraphs


##facebook: N = 4040, k = 44
N = 4040#1000
p = 0.2
num_trials = 1 #100
num_trials_perc = 200
graph_type_range = [:erdos_renyi,:watts_strogatz,:powerlaw_cluster,:fb]
graph_name_range = ["erdos renyi", "watts strogatz", "powerlaw cluster", "facebook"]
n_range = [2,7] 
k = 44#7

#histogram data
hist_all_degrees = Array(Array{Any,1},(length(n_range),length(graph_type_range)))
hist_all_thresholds = Array(Array{Any,1},(length(n_range),length(graph_type_range)))
hist_all_clustering= Array(Array{Any,1},(length(n_range),length(graph_type_range)))
for (i,n) in enumerate(n_range)
    for (j,graph_type) in enumerate(graph_type_range)
        num_trials_curr = graph_type == :fb ? 1 : num_trials
        hist_all_thresholds[i,j] = Any[]
        hist_all_degrees[i,j] = Any[]
        hist_all_clustering[i,j] = Any[]
        for l = 1:num_trials_curr
            g = create_graph(N,k,graph_type)
            p_knowns = get_p_known_percolation(g,p,n,num_trials_perc)[1]
    #         p_knowns = p_knowns[p_knowns .> 0.0]
            hist_all_degrees[i,j] = vcat(hist_all_degrees[i,j],LightGraphs.degree(g))
            hist_all_thresholds[i,j] = vcat(hist_all_thresholds[i,j],p_knowns)
            hist_all_clustering[i,j] = vcat(hist_all_clustering[i,j],LightGraphs.local_clustering_coefficient(g))
        end
        println("n = $n, graph type: $(graph_name_range[j])")

    end
end

JLD.save("../data/per_node_data/N_$(N)_$(now()).jld","hist_all_degrees",hist_all_degrees,
"hist_all_thresholds",hist_all_thresholds,"hist_all_clustering",hist_all_clustering,
"graph_type_range",graph_type_range,"graph_name_range",graph_name_range,"N",N,"k",k,
"n_range",n_range,"p",p,)
