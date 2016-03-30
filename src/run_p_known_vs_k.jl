push!(LOAD_PATH, pwd())
using LightGraphs, IndirectRec, GraphConnectivityTheory,GraphCreation
using PyCall, Distributions, JLD

N = 400
p = 0.2
num_trials = 10
num_trials_perc = 100
graph_type_range = [:watts_strogatz,:gamma_fb]#[:erdos_renyi,:watts_strogatz,:powerlaw_cluster, :gamma_fb]
graph_name_range = ["watts_strogatz", "gamma_fb"]#["erdos renyi", "watts strogatz", "powerlaw cluster", "gamma fb"]
n_range = [1,2,3,5,7,100]
k_ideal_range = collect(2:2:20)#collect(2:2:48)
c_ideal_range = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.95,0.99]
p_know_range_perc_th = zeros((length(k_ideal_range),length(graph_type_range),length(c_ideal_range)))
p_know_range_perc_order = zeros((length(k_ideal_range),length(graph_type_range),length(c_ideal_range),length(n_range)))
k_range = similar(p_know_range_perc_th)
clustering_range = similar(p_know_range_perc_th)
p_known_fn = (x,y) -> get_p_known_first_order(x,y,p)
for (q,C) in enumerate(c_ideal_range)
	for (j,graph_type) in enumerate(graph_type_range)
	    for (i,k) in enumerate(k_ideal_range)
	    	#several trials
	    	for trial_idx in 1:num_trials
		        g = create_graph(N,k,graph_type,C)
		        k_range[i,j,q] += 2*LightGraphs.ne(g)/LightGraphs.nv(g)
		        clustering_range[i,j,q] += mean(LightGraphs.local_clustering_coefficient(g))
		        p_know_range_perc_th[i,j,q] += get_p_known_percolation_theory(g,p)
		        for (l,n) in enumerate(n_range)
		            p_know_range_perc_order[i,j,q,l] += get_p_known_percolation(g,p,n,num_trials_perc)[end]
		        end
	        end
	        #renormalization
	        for (l,n) in enumerate(n_range)
	    	    p_know_range_perc_order[i,j,q,l] /= num_trials
	    	end
	        p_know_range_perc_th[i,j,q] /= num_trials
	        k_range[i,j,q] /=num_trials
	        clustering_range[i,j,q] /=num_trials
	        println("k = $k, graph type: $(graph_name_range[j])")
	    end
	end
end

JLD.save("../data/per_graph_data/N_$(N)_$(now()).jld",
"graph_type_range",graph_type_range,"graph_name_range",graph_name_range,"N",N,
"n_range",n_range,"p",p,"p_know_range_perc_th",p_know_range_perc_th,"p_know_range_perc_order",
p_know_range_perc_order,"clustering_range",clustering_range,"k_range",k_range,"num_trials",num_trials,"num_trials_perc",num_trials_perc)
