push!(LOAD_PATH, pwd())
using RandomClusteringGraph, Distributions, LightGraphs, GraphCreation
H = create_graph(4040,44,:fb)
d = fit(Pareto,degree(H))
N = 4040
C = 0.6
G = random_clustering_graph(d,N,C) 
println("Finished")
