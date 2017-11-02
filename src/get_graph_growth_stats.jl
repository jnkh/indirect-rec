push!(LOAD_PATH, pwd())
using LightGraphs, IndirectRec, GraphConnectivityTheory,GraphCreation
using PyCall, PyPlot, Distributions
using GraphPlot, CliquePercolation, StatsBase
plt[:rc]("text",usetex=true)



p = 0.5
m = 5
N = 1000
# max_degree = 00
# num_divisions = 10000000
# cache= Cache(max_degree,num_divisions)
cache2= GenCache(Dict{Tuple{Int,Float64,Int,Float64},Float64}())
# Profile.clear()
ii = unique(Int64.(floor.(logspace(0,log10(N),10))))
m_range = [15]#,7,20]
graph_types = [:ff,:n,:P,:k] #:P
markers = ["o","^","s","<","o"]
colors_range = ["b","g","r","k"]
labels = ["n_{i,j,approx}","n_{ij}","P_{ij}","k_{i}","dummy","dummy"]
linestyles = ["-","--","-.","-","-"]
outputs = [] 
num_trials = 1
for symb in graph_types
# symb = :n
# for m in m_range
# for symb in graph_types
    new_output = []
    @time for i = 1:num_trials
        g,C_vs_N,L_vs_N = build_graph(m,N,p,cache2,ii,symb)
#         g,C_vs_N,L_vs_N = generate_trust_graph_2(N,m,ii)
        ks = degree(g)
        cs = local_clustering_coefficient(g)
        ns = (ks-1).*cs
        if length(new_output) == 0
            new_output = Any[C_vs_N,L_vs_N,ks,cs,ns,[g]]
        else
            new_output[1] += C_vs_N
            new_output[2] += L_vs_N
            new_output[3] = vcat(new_output[3],ks)
            new_output[4] = vcat(new_output[4],cs)
            new_output[5] = vcat(new_output[5],ns)
            push!(new_output[6],g)
        end
    end
    new_output[1] = Array{Float64}(new_output[1])
    new_output[2] = Array{Float64}(new_output[2])
    new_output[3] = Array{Int}(new_output[3])
    new_output[4] = Array{Float64}(new_output[4])
    new_output[5] = Array{Float64}(new_output[5])
    new_output[1] /= num_trials
    new_output[2] /= num_trials
    push!(outputs,new_output)
end

using JLD
save("../data/graph_generation/growth_outputs_N_$(N)_m_$(m)_$(now()).jld","outputs",outputs,"graph_types",graph_types,"N",N,
