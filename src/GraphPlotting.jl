module GraphPlotting

using Colors, StatsBase, GraphPlot
using LightGraphs, IndirectRec, GraphConnectivityTheory,GraphCreation
using PyCall, PyPlot, Distributions
using CliquePercolation

export plot_graph_colored_by_p_th, plot_graph_colored_by_p_sim,
plot_P_graph,plot_graph

using StatsBase, Colors

function scale_to_indices(arr,color_range=nothing)
    if color_range == nothing
        maxval = maximum(arr)
        minval = minimum(arr)
    else
        minval = color_range[1]
        maxval = color_range[2]
    end

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

function get_colors_from_array(arr,cm,by_rank,color_range)
    println(by_rank)
    println(color_range)
    if by_rank
        indices = scale_to_indices_rank(arr)
    else
        indices = scale_to_indices(arr,color_range)
    end
    cols = [cm[indices[i]] for i in 1:length(arr)]
    return cols
end

function rgb_to_rgba(val,alpha)
    ret = RGBA(val.r,val.g,val.b,alpha)
end

function get_edge_colors(g,p,cm,alpha,rho_prime,by_rank,color_range)
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
    edgecolor =  rgb_to_rgba.(get_colors_from_array(ps,cm,by_rank,color_range),alpha)
    edgecolorth =  rgb_to_rgba.(get_colors_from_array(ps_th,cm,by_rank,color_range),alpha)
    return edgecolor,edgecolorth
end

function plot_graph_colored_by_p_sim(g,locx,locy,p,trials=400,alpha=0.5,by_rank=true,color_range=nothing)
    # g = GraphCreation.create_graph(N,k,graph_type,C)
    ks = degree(g)
    cs = local_clustering_coefficient(g)
    ns = (ks-1).*cs

    ps,_ = get_p_known_clique_percolation(g,p,trials)
    cm = colormap("RdBu")
    cols = rgb_to_rgba.(get_colors_from_array(ps,cm,by_rank,color_range),alpha)
    NODESIZE = 0.05
    edgecolor = RGBA(0.0,0.0,0.0,0.3)

    edgec,edgecth = get_edge_colors(g,p,cm,1.0,0.5,by_rank,color_range);
    gplot(g,locx,locy,nodesize=1+ns,nodefillc=cols,
    NODESIZE=NODESIZE,nodestrokec=colorant"black",
    nodestrokelw=2,edgestrokec=edgec)#,nodelabel=ks)
end

function plot_graph_colored_by_p_th(g,locx,locy,p,trials=400,alpha=0.5,by_rank=true,color_range=nothing)
    # g = GraphCreation.create_graph(N,k,graph_type,C)
    ks = degree(g)
    cs = local_clustering_coefficient(g)
    ns = (ks-1).*cs

    ps_th = Float64.(get_p_known_clique_theory.(ks,cs,p))
    cm = colormap("RdBu")
    cols_th = rgb_to_rgba.(get_colors_from_array(ps_th,cm,by_rank,color_range),alpha)
    NODESIZE = 0.05
    edgecolor = RGBA(0.0,0.0,0.0,0.3)

    edgec,edgecth = get_edge_colors(g,p,cm,1.0,0.5,by_rank,color_range);
    gplot(g,locx,locy,nodesize=1+ns,nodefillc=cols_th,
    NODESIZE=NODESIZE,nodestrokec=colorant"black",
    nodestrokelw=2,edgestrokec=edgecth)#,nodelabel=ks)
end

function plot_P_graph(g,p,N;color_range=[0.0,1.0],color_range_edge=nothing,
    edge_color="grey",cm="inferno",as_neighborhood=false,
    lw=2,fac=10.0,num_trials=1000,show_colorbar=true,edge_alpha=1.0,
    alpha=1.0,log_scaling=true,layout_fn=nothing,theory=true)
#     locx,locy = random_layout(g)
    #     p_knowns = get_p_known_percolation(g,p,N,num_trials)[1]
    if layout_fn == nothing
        layout_fn = spring_layout
    end
    if color_range_edge == nothing
        color_range_edge = color_range
    end

    if as_neighborhood
        add_complete_vertex(g)
        v = nv(g)
        p_knowns = Float64[]
        for w in vertices(g)[1:end-1]
            if !theory
                t = degree(g,v)*get_p_known_from_neighbor_to_other_neighbor(g,p,v,w,num_trials)
            else
                t = get_edge_critical_thresh_theory(g,v,w,p)
            end
#             t2 = get_edge_critical_thresh_theory(g,w,v,p)
#             t = min(t1,t2)
            push!(p_knowns,t)
        end
#         p_known_tot = get_node_critical_thresh_theory(g,v,p)
        p_known_tot = get_p_known_percolation(g,p,nv(g),num_trials)[1][v]
        p_known_error = get_binomial_error(p_known_tot,num_trials)
        rem_vertex!(g,nv(g))
    else
        p_knowns = get_node_critical_thresh_theory.(g,vertices(g),p)
    end
    locx,locy = layout_fn(g)
    ks = degree(g)
    cs = local_clustering_coefficient(g)
    ns = (ks-1).*cs
    println("P: $(@sprintf("%.3f",minimum(p_knowns))), $(@sprintf("%.3f",mean(p_knowns))), $(@sprintf("%.3f",maximum(p_knowns)))")
    if color_range == nothing
        vmin = minimum(p_knowns)
        vmax = maximum(p_knowns)
    else
        vmin = color_range[1]
        vmax = color_range[2]
    end

    scatter(locx,locy,s=fac*(0.2+ks),c=p_knowns,vmin=vmin,vmax=vmax,cmap=cm,alpha=alpha,linewidth=0)
    if show_colorbar
        colorbar()
    end
    
    
    edge_pos = [((locx[e.src],locy[e.src]),(locx[e.dst],locy[e.dst])) for e in edges(g)]
    P_tildes = Float64[]
    for (i,e) in enumerate(edges(g))
        t1 = get_edge_critical_thresh_theory(g,e.src,e.dst,p)
        t2 = get_edge_critical_thresh_theory(g,e.dst,e.src,p)
        t = min(t1,t2)
        push!(P_tildes,t)
    end
    if(log_scaling)
        P_tildes = log10(P_tildes - p+1e-2)
    end
    println("P_tilde: $(@sprintf("%.3f",minimum(P_tildes))), $(@sprintf("%.3f",mean(P_tildes))), $(@sprintf("%.3f",maximum(P_tildes)))")
        
    edge_collection = matplotlib[:collections][:LineCollection](edge_pos,alpha=edge_alpha,
    colors=edge_color,linewidths=lw,antialiaseds=(1,),linestyle="-")
    edge_collection[:set_zorder](0)  # edges go behind nodes
    if !as_neighborhood
        edge_collection[:set_array](P_tildes)
    else
        title(latexstring("\$P = $(@sprintf("%.4f",p_known_tot)) \\pm $(@sprintf("%.4f",p_known_error))\$"))
    end
    edge_collection[:set_cmap](cm)
    if color_range_edge == nothing
        edge_collection[:autoscale]()
    else
        edge_collection[:set_clim](color_range_edge[1],color_range_edge[2])
    end
    gca()[:add_collection](edge_collection)
    if show_colorbar
        colorbar()
    end
    zoom = 1.5
    xlim([-1.0*zoom,1.0*zoom])
    ylim([-1.0*zoom,1.0*zoom])
    axis("off")
end


function plot_graph(g,N;node_color="grey",edge_color="grey"
    ,lw=2,fac=10.0,edge_alpha=1.0,alpha=1.0,layout_fn=nothing,
    zoom = 1.5)
#     locx,locy = random_layout(g)
    #     p_knowns = get_p_known_percolation(g,p,N,num_trials)[1]
    if layout_fn == nothing
        layout_fn = spring_layout
    end


    locx,locy = layout_fn(g)
    ks = degree(g)
    cs = local_clustering_coefficient(g)
    ns = (ks-1).*cs

    scatter(locx,locy,s=fac*(0.2+ks),c=node_color,alpha=alpha,linewidth=0)
    
    edge_pos = [((locx[e.src],locy[e.src]),(locx[e.dst],locy[e.dst])) for e in edges(g)]

        
    edge_collection = matplotlib[:collections][:LineCollection](edge_pos,alpha=edge_alpha,
    colors=edge_color,linewidths=lw,antialiaseds=(1,),linestyle="-")
    edge_collection[:set_zorder](0)  # edges go behind nodes

    gca()[:add_collection](edge_collection)
    xlim([-1.0*zoom,1.0*zoom])
    ylim([-1.0*zoom,1.0*zoom])
    axis("off")
end




end
