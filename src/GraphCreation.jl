module GraphCreation

using PyCall, LightGraphs, Distributions,StatsBase,CliquePercolation
using RandomClusteringGraph, IndirectRec

export create_graph, networkx_to_lightgraph,
 read_edgelist, read_edgelist_julia,
 generate_gaussian_graph,add_complete_vertex,
 build_graph,produce_clique_graph_ring,GenCache,
 generate_trust_graph,generate_trust_graph_old,
 subsample_graph

function create_graph(N,k,graph_type=:erdos_renyi,clustering=0.6;deg_distr=nothing)
    p_edge = k/(N-1)
    if graph_type == :erdos_renyi
        g = LightGraphs.erdos_renyi(N,p_edge)
    elseif graph_type == :watts_strogatz
        g = watts_strogatz_with_custering(N,Int(round(k)),clustering)
    elseif graph_type == :random_regular
        g = LightGraphs.random_regular_graph(N,Int(round(k)))
    elseif graph_type == :powerlaw_cluster
        g = powerlaw_cluster_graph(N,Int(round(0.5*(N-1)*p_edge)),1.0)
    elseif graph_type == :geometric_gauss
        g,_,_ = generate_gaussian_graph(N,k)
    elseif graph_type == :fb
        g = read_edgelist("../data/graphs/facebook_combined.txt")
    elseif graph_type == :fb_gamma
        g = read_edgelist("../data/graphs/fb_to_gamma_fit_clustering.dat")
    elseif graph_type == :fb_normal
        g = read_edgelist("../data/graphs/fb_to_normal_fit_clustering.dat")
    elseif graph_type == :fb_bter
        g = read_edgelist("../data/graphs/bter_fit_facebook.dat")
    elseif graph_type == :gamma_fb
        g = random_clustering_graph(Gamma(1.0,k),N,clustering)
    elseif graph_type == :rand_clust
        if deg_distr == nothing
            deg_distr = Gamma(1.0,k)
        end
        g = random_clustering_graph(deg_distr,N,clustering)
    else
        error("invalid graph type")
    end
    g
end

function subsample_graph(g,N_new)
    g1 = induced_subgraph(g,sample(vertices(g),N_new,replace=false))[1]
    comp = sort(connected_components(g1),by=length)[end]
    g2 = induced_subgraph(g1,comp)[1]
    return g2
end

#helper function
function add_complete_vertex(h)
    add_vertex!(h)
    vert = vertices(h)
    new_v = length(vert)
    for w in vert 
        if new_v != w
            add_edge!(h,new_v,w)
        end
    end
    h
end


###RANDOM GEOMETRIC GRAPH
function getGaussianConstant(n,s,k)
    return ((pi/k)*(n/(s*s)))
end

function periodic_distance(locx,locy,i,j,length)
    dx = periodic_dist_one_dim(locx[i],locx[j],length)
    dy = periodic_dist_one_dim(locy[i],locy[j],length)
    return sqrt(dx^2 + dy^2)
end


    
function periodic_dist_one_dim(x1,x2,length)
    d = abs(x1-x2)
    return min(d,length-d)
end

function checkEdge!(g,i,j,a,locx,locy,length)
    #d = distance([locx[i],locy[i]],[locx[j],locy[j]],length)
    d = periodic_distance(locx,locy,i,j,length)
    gaussianProbability = e^(-a*d^2)
    r = rand()
    if (r<gaussianProbability)
            add_edge!(g,i,j)
            return true
            #println("d=($d),  probability=$gaussianProbability, r = $r   Edge added")
    end
    return false
end   

function generate_gaussian_graph(N,k,locx=nothing,locy=nothing)
    GRAPH_LOCATION_SIZE = 1
    if locx == nothing || locy == nothing
        locx = rand(N)*GRAPH_LOCATION_SIZE;
        locy = rand(N)*GRAPH_LOCATION_SIZE;
    end

    g = Graph(N)
    alpha = getGaussianConstant(N,GRAPH_LOCATION_SIZE,k)
    println("length scale: $(1/sqrt(alpha))")
    for i in LightGraphs.vertices(g)
        for j in LightGraphs.vertices(g)
            if j > i
                checkEdge!(g,i,j,alpha,locx,locy,GRAPH_LOCATION_SIZE)
            end
        end
    end
    #k_meas = mean(degree(g))
    #c_meas = mean(local_clustering_coefficient(g))
    #println("degree = $(k_meas), clustering = $(c_meas)")
    return g,locx,locy
end


###WATTS_STROGATZ###
function watts_strogatz_get_clustering(N,K,beta)
    c1,c0 = watts_strogatz_clustering_limits(N,K)
    return c1 + (c0-c1)*(1-beta)^3
end

function watts_strogatz_get_beta(N,K,C)
    c1,c0 = watts_strogatz_clustering_limits(N,K)
    beta = 1 - ((C - c1)/(c0 - c1))^(1.0/3)
    return beta
end

function watts_strogatz_clustering_limits(N,K)
    c1 = K/N
    c0 = 3*(K-2)/(4*(K-1))
    return c1,c0
end

function watts_strogatz_with_custering(N,K,C)
    c1,c0 = watts_strogatz_clustering_limits(N,K)
    C = clamp(C,c1,c0)
    beta = watts_strogatz_get_beta(N,K,C)
    return LightGraphs.watts_strogatz(N,K,beta)
end

###NETWORKX###
@pyimport networkx as nx
function networkx_to_lightgraph(G)
    g = LightGraphs.Graph(length(nx.nodes(G)))
    nx_nodes = convert(Array{Int,1},nx.nodes(G))
    nx_to_lg = get_mapping(nx_nodes)
    for e in nx.edges(G)
        LightGraphs.add_edge!(g,nx_to_lg[e[1]],nx_to_lg[e[2]])
    end
    g
end


function get_mapping(nx_nodes::Array{Int,1})
    nx_to_lg = Dict{Int,Int}()
    for (lg_node,nx_node) in enumerate(nx_nodes)
        nx_to_lg[nx_node] = lg_node
    end
    return nx_to_lg
end




function powerlaw_cluster_graph(N::Int,k::Int,beta::Real)
    G = nx.powerlaw_cluster_graph(N,k,beta)
    return networkx_to_lightgraph(G)
end


function make_networkx_graph_from_lightgraph(G::LightGraphs.Graph)
    H = nx.empty_graph(length(LightGraphs.vertices(G)))
    for e in LightGraphs.edges(G)
        H[:add_edge](e[1]-1,e[2]-1)
    end
    H
end

function read_edgelist(filename)
    curr_edges = readdlm(filename,Int)
    curr_nodes = unique(curr_edges)
    g = LightGraphs.Graph(length(curr_nodes))
    to_lg = get_mapping(curr_nodes)
    for i = 1:size(curr_edges,1)
        LightGraphs.add_edge!(g,to_lg[curr_edges[i,1]],to_lg[curr_edges[i,2]])
    end
    g
end

function read_edgelist_nx(filename)
    G = nx.read_edgelist(filename,nodetype=int)
    return networkx_to_lightgraph(G)
end



function build_graph(m,N,p,cache,log_points=nothing,attachment_mode=:P,mu=nothing)
    if attachment_mode == :ff
        return generate_trust_graph(N,m,log_points)
    end
    g = Graph()
    C_vs_N = Float64[]
    L_vs_N = Float64[]
    if log_points != nothing
        ii = log_points
    else
        ii = unique(Int64.(floor.(logspace(0,log10(N),20))))
    end
    for i = 1:N
        if i <= m
            add_complete_vertex(g)
        else
            add_sampled_vertex(g,m,p,cache,attachment_mode,mu)
        end
        if i in ii
            push!(C_vs_N,mean(local_clustering_coefficient(g)))
            push!(L_vs_N,sum(floyd_warshall_shortest_paths(g).dists)/(i*(i-1)))
        end
    end
    g,C_vs_N,L_vs_N
end

function rewire_random_edge(g)
    v = sample(vertices(g))
    num_neighbors = length(neighbors(g,v))
    while  num_neighbors == 0 || num_neighbors > nv(g) - 1
        v = sample(vertices(g))
    end
    w = sample(vertices(g))
    while w in neighbors(g,v)
        w = sample(vertices(g))
    end
    w_old = sample(neighbors(g,v))
    rem_edge!(g,v,w_old)
    add_edge!(g,v,w)
end
    

function add_sampled_vertex(g,m,p,cache,attachment_mode,mu)
    old_vertices = collect(vertices(g))
    assert(length(old_vertices) >= m)
    add_vertex!(g)
    v = nv(g)
    for j = 1:m
        num_vertices = length(old_vertices)
#         weights = ones(num_vertices)
        if attachment_mode == :P 
            weights = get_sample_weights(g,old_vertices,v,p,cache,mu)
        elseif attachment_mode == :n
            weights = get_sample_weights_approx(g,old_vertices,v,p,mu)
        elseif attachment_mode == :k
            ks = degree(g)
            cs = local_clustering_coefficient(g)
#             weights = ((ks-1).*cs)[old_vertices]
            weights = ks[old_vertices]
        elseif attachment_mode == :rand
            weights = ones(length(old_vertices))
        else
            println("Invalid attachment mode $(attachment_mode)")
            return
        end
        w = sample(old_vertices,Weights(weights))
        assert(add_edge!(g,v,w))
        idx = searchsorted(old_vertices,w)[1]
        assert(old_vertices[idx] == w)
        deleteat!(old_vertices,idx)
    end
end


function get_sample_weights_approx(g,vs,v,p,mu_fn = nothing)
    if mu_fn == nothing
        mu_fn = x -> x
    end
    weights = zeros(length(vs))
    trials = 100
    for (i,w) in enumerate(vs)
        assert(add_edge!(g,v,w))
        k = degree(g,w)
        
#         c = local_clustering_coefficient(g,w)
        n = get_num_mutual_neighbors(g,Edge(w,v))
        if k > 0
            ret = n
        else
            ret = 0
        end
        
        weights[i] = mu_fn(ret) 
        rem_edge!(g,v,w)
    end
    if sum(weights) == 0
        weights += 0.1
    end
    return weights
end

function get_sample_weights(g,vs,v,p,cache,mu_fn=nothing)
    if mu_fn == nothing
        mu_fn = x -> x
    end
    weights = zeros(length(vs))
    trials = 100
    for (i,w) in enumerate(vs)
        assert(add_edge!(g,v,w))
        k = degree(g,w)
        
        c = local_clustering_coefficient(g,w)
        n = get_num_mutual_neighbors(g,Edge(w,v))
        P_tilde_th = get_p_known_clique_neighbor_to_neighbor_theory_fast(k,c,n,p,cache)
#             P_tilde_th = get_p_known_clique_neighbor_to_neighbor_theory(k,c,n,p)
        ret = P_tilde_th
#         println(abs(P_tilde_th - P_tilde))#/P_tilde_th)
        

        assert((k*ret)-p >= -1e-12)
#         println("total: $(abs(k*ret - p - p^2*n))")

#         weights[i] = max(0,(k*ret)-p )^mu
        weights[i] = mu_fn(max(0,(k*ret)-p ))
        rem_edge!(g,v,w)
    end
    if sum(weights) == 0
        weights += 0.1
    end
    return weights
end


function get_uniform_sample_weights(g,vs,v,p)
#     return local_clustering_coefficient(g)[vs].*(degree(g)[vs]-1).*(degree(g)[vs])
    return (degree(g)[vs])
#     return ones(length(vs))
end


type GenCache{K,V}
    cache::Dict{K,V}
end

function GenCache{K,V}(d::Dict{K,V})
    return GenCache{K,V}(d)
end
    

function set_value{K,V}(c::GenCache{K,V},x::K,y::V)
#     println("recomputed $(x)")
    c.cache[x] = y 
end

function get_value{K,V}(c::GenCache{K,V},x::K)
    return c.cache[x]
end

function has_key{K,V}(c::GenCache{K,V},x::K)
    return haskey(c.cache,x)
end


function get_p_known_clique_neighbor_to_neighbor_theory_fast(k::Int,c::Float64,n::Int,p::Float64,cache::GenCache)
    if has_key(cache,(k,c,n,p))
        return get_value(cache,(k,c,n,p))
    end
    P_other_neighbors = 0.0
    if n > 0
        
        c1 = (c*(k*(k-1))/2-n)/((k-1)*(k-2)/2)
#         P_other_neighbors = Float64(get_p_known_clique_theory_fast(k-1,c1*(k-1)/n,n/(k-1)*p,cache,cache2))
        P_other_neighbors = Float64(get_p_known_clique_theory(k-1,c1*(k-1)/n,n/(k-1)*p))
    end

    ret = p/k*(1 + (k-1)*P_other_neighbors)
    if ret < p/k
        println(P_other_neighbors)
    end
    set_value(cache,(k,c,n,p),ret)
    return ret
end

function produce_clique_graph_ring(N)
    h = Graph(N)
    for i = 1:N-1
        add_edge!(h,Edge(i,i+1))
    end
    add_edge!(h,Edge(N,1))
    println(mean(degree(h)))

    add_complete_vertex(h)
end

# P \sim n^\mu



function generate_trust_graph(N,m,log_points=nothing)
    #start with complete graph
    C_vs_N = Float64[]
    L_vs_N = Float64[]
    if log_points != nothing
        ii = log_points
    else
        ii = unique(Int64.(floor.(logspace(0,log10(N),20))))
    end
    
    g = Graph()
    N_curr = 0
    weights = zeros(Int,N)
    #add a node
    for i = 1:N
        if i <= m+1
            add_complete_vertex(g)
        else
            add_vertex!(g)
            new_v = N_curr+1
            v = sample(1:N_curr,Weights(degree(g)[1:N_curr]))
#             v = sample(1:N_curr)
            assert(add_edge!(g,N_curr+1,v))
            #add other edges
            for j = 2:m
                v = sample(neighbors(g,new_v))
                w = sample(neighbors(g,v))
                while w == new_v || !add_edge!(g,new_v,w) 
                    v = sample(neighbors(g,new_v))
                    w = sample(neighbors(g,v))
                end
            end
        end
        N_curr += 1
        if i in ii
            push!(C_vs_N,mean(local_clustering_coefficient(g)))
            push!(L_vs_N,sum(floyd_warshall_shortest_paths(g).dists)/(i*(i-1)))
            # push!(L_vs_N,1.0)#sum(floyd_warshall_shortest_paths(g).dists)/(i*(i-1)))
        end
    end
    

    g,C_vs_N,L_vs_N
    
end



#much faster!

function generate_trust_graph_old(N,m,log_points=nothing)
    #start with complete graph
    C_vs_N = Float64[]
    L_vs_N = Float64[]
    if log_points != nothing
        ii = log_points
    else
        ii = unique(Int64.(floor.(logspace(0,log10(N),20))))
    end
    
    g = Graph()
    N_curr = 0
    weights = zeros(Int,N)
    #add a node
    for i = 1:N
        if i <= m+1
            add_complete_vertex(g)
        else
            weights *= 0
            add_vertex!(g)
#             v = sample(1:N_curr)
            v = sample(1:N_curr,Weights(degree(g)[1:N_curr]))
            assert(add_edge!(g,N_curr+1,v))
            add_neighbors_to_array(weights,g,v)
            #add other edges
            for j = 2:m
                v = sample(1:N_curr,Weights(weights[1:N_curr]))
                assert(sum(weights[N_curr+1:end]) == 0)
                assert(add_edge!(g,N_curr+1,v))
                assert(v != N_curr+1)
                add_neighbors_to_array(weights,g,v)
                weights[v] = 0
            end
        end
        N_curr += 1
        if i in ii
            push!(C_vs_N,mean(local_clustering_coefficient(g)))
            push!(L_vs_N,sum(floyd_warshall_shortest_paths(g).dists)/(i*(i-1)))
#             push!(L_vs_N,1.0)#sum(floyd_warshall_shortest_paths(g).dists)/(i*(i-1)))
        end
    end
    

    g,C_vs_N,L_vs_N
    
end

function add_neighbors_to_array(vec::Array{Int,1},g,v)
    vs = neighbors(g,v)
    last_node = nv(g)
    for w in vs 
        if w != last_node
            vec[w] += 1
        end
    end
    for w in neighbors(g,last_node)
        vec[w] = 0
    end
end
       







end
