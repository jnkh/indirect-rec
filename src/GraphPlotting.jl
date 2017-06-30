module GraphPlotting

using Colors, StatsBase, GraphPlot
using LightGraphs, IndirectRec, GraphConnectivityTheory,GraphCreation
using PyCall, PyPlot, Distributions
using CliquePercolation

export plot_graph_colored_by_p_th, plot_graph_colored_by_p_sim

using StatsBase, Colors

function scale_to_indices(arr)
    maxval = maximum(arr)
    minval = minimum(arr)
    if maxval == minval
        arr =  0.5*ones(arr)
    else
        arr = (arr - minval)/(maxval-minval)
    end
    arr = clamp.(arr*100,1,100)
    arr = Int.(round.(arr))
    return arr
end

function scale_to_indices_rank(arr)
    ranks = tiedrank(arr)
    return scale_to_indices(ranks)
end

function get_colors_from_array(arr,cm)
#     indices = scale_to_indices(arr)
    indices = scale_to_indices_rank(arr)
    cols = [cm[indices[i]] for i in 1:length(arr)]
    return cols
end

function rgb_to_rgba(val,alpha)
    ret = RGBA(val.r,val.g,val.b,alpha)
end

function get_edge_colors(g,p,cm,alpha,rho_prime = 0.5)
    trials = 100
    ps = Float64[]
    ps_th = Float64[]
    for e in edges(g)
        v = e.dst
        w = e.src
        k = degree(g,v)
        k2 = degree(g,w)
        c = local_clustering_coefficient(g,v)
        c2 = local_clustering_coefficient(g,w)
        n = get_num_mutual_neighbors(g,Edge(v,w))
        
        ps1 = get_p_known_from_neighbor_to_other_neighbor(g,p,v,w,trials)
        ps2 = get_p_known_from_neighbor_to_other_neighbor(g,p,w,v,trials)
        c_b_1 = 1/edgewise_critical_b_c(rho_prime,k,ps1)
        c_b_2 = 1/edgewise_critical_b_c(rho_prime,k2,ps2)
        push!(ps,min(c_b_1,c_b_2))
        
        ps_th1 = Float64(get_p_known_clique_neighbor_to_neighbor_theory(k,c,n,p))
        ps_th2 = Float64(get_p_known_clique_neighbor_to_neighbor_theory(k2,c2,n,p))
        c_b_1 = 1/edgewise_critical_b_c(rho_prime,k,ps_th1)
        c_b_2 = 1/edgewise_critical_b_c(rho_prime,k2,ps_th2)
        push!(ps_th,min(c_b_1,c_b_2))  #plot min because edge will get cut if at least one of the values is small
    end
    edgecolor =  rgb_to_rgba.(get_colors_from_array(ps,cm),alpha)
    edgecolorth =  rgb_to_rgba.(get_colors_from_array(ps_th,cm),alpha)
    return edgecolor,edgecolorth
end

function plot_graph_colored_by_p_sim(g,locx,locy,p,trials=400,alpha=0.5)
    # g = GraphCreation.create_graph(N,k,graph_type,C)
    ks = degree(g)
    cs = local_clustering_coefficient(g)
    ns = (ks-1).*cs

    ps,_ = get_p_known_clique_percolation(g,p,trials)
    cm = colormap("RdBu")
    cols = rgb_to_rgba.(get_colors_from_array(ps,cm),alpha)
    NODESIZE = 0.05
    edgecolor = RGBA(0.0,0.0,0.0,0.3)

    edgec,edgecth = get_edge_colors(g,p,cm,1.0);
    gplot(g,locx,locy,nodesize=1+ns,nodefillc=cols,
    NODESIZE=NODESIZE,nodestrokec=colorant"black",
    nodestrokelw=2,edgestrokec=edgec)#,nodelabel=ks)
end

function plot_graph_colored_by_p_th(g,locx,locy,p,trials=400,alpha=0.5)
    # g = GraphCreation.create_graph(N,k,graph_type,C)
    ks = degree(g)
    cs = local_clustering_coefficient(g)
    ns = (ks-1).*cs

    ps_th = Float64.(get_p_known_clique_theory.(ks,cs,p))
    cm = colormap("RdBu")
    cols_th = rgb_to_rgba.(get_colors_from_array(ps_th,cm),alpha)
    NODESIZE = 0.05
    edgecolor = RGBA(0.0,0.0,0.0,0.3)

    edgec,edgecth = get_edge_colors(g,p,cm,1.0);
    gplot(g,locx,locy,nodesize=1+ns,nodefillc=cols_th,
    NODESIZE=NODESIZE,nodestrokec=colorant"black",
    nodestrokelw=2,edgestrokec=edgecth)#,nodelabel=ks)
end


end
