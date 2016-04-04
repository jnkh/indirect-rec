push!(LOAD_PATH, pwd())
using RandomClusteringGraph, Distributions, LightGraphs, GraphCreation
H = create_graph(4040,44,:fb)
d = fit(Gamma,degree(H))
N = 4040
C = 0.79
G = random_clustering_graph(d,N,C,false,"fb_to_gamma_fit_clustering.dat")
println("Finished")
