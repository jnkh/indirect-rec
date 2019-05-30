module GraphCreation

using Random,PyCall, LightGraphs, Distributions,StatsBase,CliquePercolation
using RandomClusteringGraph, IndirectRec
import DelimitedFiles

export create_graph, networkx_to_lightgraph,
 read_edgelist, read_edgelist_julia,
 generate_gaussian_graph,add_complete_vertex,
 build_graph,build_graph_watts_strogatz,produce_clique_graph_ring,GenCache,
 generate_trust_graph,generate_trust_graph_old,
 subsample_graph,sample_connected_subgraph,
 sample_subgraph_by_community_given_k,
 subsample_from_communities

function create_graph(N,k,graph_type=:erdos_renyi,clustering=0.5;deg_distr=nothing)
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
    elseif graph_type in [:ff,:ffg,:k,:P,:Psim,:n]
        cache= GenCache(Dict{Tuple{Int,Float64,Int,Float64},Float64}())
        p = 0.1
        return build_graph(k/2.0,N,p,cache,[1],graph_type,nothing,C=clustering)[1]
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
    gaussianProbability = exp(-a*d^2)
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
nx = pyimport("networkx")
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
    curr_edges = DelimitedFiles.readdlm(filename,Int)
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

function build_graph_watts_strogatz(N,m,C,log_points)
    k = 2*m
    C_vs_N = Float64[]
    L_vs_N = Float64[]
    g_curr = nothing
    for N_curr = log_points
        @assert(N_curr*k % 2 == 0)
        g_curr = create_graph(N_curr,k,:watts_strogatz,C)
        push!(C_vs_N,mean(local_clustering_coefficient(g_curr)))
        push!(L_vs_N,sum(floyd_warshall_shortest_paths(g_curr).dists)/(N_curr*(N_curr-1)))
    end
    return g_curr,C_vs_N,L_vs_N
end

function build_graph_fb(N,m,C,log_points)
    k = 2*m
    C_vs_N = Float64[]
    L_vs_N = Float64[]
    g_curr = nothing
    g_orig = create_graph(N,k,:fb,C)
    for N_curr = log_points
        @assert(N_curr*k % 2 == 0)
        g_curr = subsample_from_communities(g_orig,N_curr,k)
        push!(C_vs_N,mean(local_clustering_coefficient(g_curr)))
        push!(L_vs_N,sum(floyd_warshall_shortest_paths(g_curr).dists)/(N_curr*(N_curr-1)))
    end
    return g_curr,C_vs_N,L_vs_N
end

function build_graph(m,N,p,cache,log_points=nothing,attachment_mode=:P,mu=nothing;C=0.5,trials=100)
    if attachment_mode == :ffg
        return generate_trust_graph_geometric(N,m,log_points)
    end
    m = Int(round(m))
    if attachment_mode == :ff
        return generate_trust_graph(N,m,log_points)
    end

    g = Graph()
    C_vs_N = Float64[]
    L_vs_N = Float64[]
    if log_points != nothing
        ii = log_points
    else
        ii = unique(Int64.(floor.(logspace(log10(2*m+1),log10(N),20))))
    end
    if attachment_mode == :watts_strogatz
        return build_graph_watts_strogatz(N,m,C,ii)
    elseif attachment_mode == :fb
        return build_graph_fb(N,m,C,ii)
    end
    for i = 1:N
        if i <= 2*m+1
            add_complete_vertex(g)
        else
            add_sampled_vertex(g,m,p,cache,attachment_mode,mu,trials)
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
    

function add_sampled_vertex(g,m,p,cache,attachment_mode,mu,trials)
    old_vertices = collect(vertices(g))
    @assert(length(old_vertices) >= m)
    add_vertex!(g)
    v = nv(g)
    for j = 1:m
        num_vertices = length(old_vertices)
#         weights = ones(num_vertices)
        if attachment_mode == :Psim 
            weights = get_sample_weights(g,old_vertices,v,p,cache,mu,theory=false,num_trials=trials)
        elseif attachment_mode == :P 
            weights = get_sample_weights(g,old_vertices,v,p,cache,mu,theory=true,num_trials=trials)
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
        @assert(add_edge!(g,v,w))
        idx = searchsorted(old_vertices,w)[1]
        @assert(old_vertices[idx] == w)
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
        @assert(add_edge!(g,v,w))
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

function get_sample_weights(g,vs,v,p,cache,mu_fn=nothing;theory=true,num_trials=100)
    if mu_fn == nothing
        mu_fn = x -> x
    end
    weights = zeros(length(vs))
    for (i,w) in enumerate(vs)
        @assert(add_edge!(g,v,w))
        k = degree(g,w)
        
        c = local_clustering_coefficient(g,w)
        n = get_num_mutual_neighbors(g,Edge(w,v))
        if n > 0
            if theory
                if k > 150
                    ret = get_audience(g,p,w,v,num_trials) #from w to v!
                else
                    ret = get_audience_theory_fast(k,c,n,p,cache)
                end
            else
                ret = get_audience(g,p,w,v,num_trials) #from w to v!
            end
            # P_tilde_th = get_p_known_clique_neighbor_to_neighbor_theory_fast(k,c,n,p,cache)
    #             P_tilde_th = get_p_known_clique_neighbor_to_neighbor_theory(k,c,n,p)
            # ret = P_tilde_th
    #         println(abs(P_tilde_th - P_tilde))#/P_tilde_th)
            

            if ret < -1e-12
                println("nij value negative: $(k*ret - p)")
                println("k: $k, c: $c, n: $n, p: $p")
            end

    #         weights[i] = max(0,(k*ret)-p )^mu
            weights[i] = mu_fn(max(0,ret))
        else
            weights[i] = 0.0
        end
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


struct GenCache{K,V}
    cache::Dict{K,V}
end

function GenCache(d::Dict{K,V}) where {K,V}
    return GenCache{K,V}(d)
end
    

function set_value(c::GenCache{K,V},x::K,y::V) where {K,V}
#     println("recomputed $(x)")
    c.cache[x] = y 
end

function get_value(c::GenCache{K,V},x::K) where {K,V}
    return c.cache[x]
end

function has_key(c::GenCache{K,V},x::K) where {K,V}
    return haskey(c.cache,x)
end

function get_audience_theory_fast(k::Int,c::Float64,n::Int,p::Float64,cache::GenCache)
    if has_key(cache,(k,c,n,p))
        return get_value(cache,(k,c,n,p))
    end
    ret = get_audience_theory(k,c,n,p)
    if ret < 0
        println(P_other_neighbors)
    end
    set_value(cache,(k,c,n,p),ret)
    return ret
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



function generate_trust_graph_geometric(N,m,log_points=nothing)
    @assert(m > 1)
    p = min(1.0,1.0/(m))
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
        if i <= 2*m+1
            add_complete_vertex(g)
        else
            j = 0
            add_vertex!(g)
            new_v = N_curr+1
            while true
                if j == 0
                    v = sample(1:N_curr,Weights(degree(g)[1:N_curr]))
        #             v = sample(1:N_curr)
                    @assert(add_edge!(g,N_curr+1,v))
                else
                #add other edges
                    v = sample(neighbors(g,new_v))
                    w = sample(neighbors(g,v))
                    while w == new_v || !add_edge!(g,new_v,w) 
                        v = sample(neighbors(g,new_v))
                        w = sample(neighbors(g,v))
                    end
                end
                j += 1
                if rand() < p || j >= i-1
                    break
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




function generate_trust_graph(N,m,log_points=nothing)
    #start with complete graph
    m = Int(round(m))
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
        if i <= 2*m+1
            add_complete_vertex(g)
        else
            add_vertex!(g)
            new_v = N_curr+1
            v = sample(1:N_curr,Weights(degree(g)[1:N_curr]))
#             v = sample(1:N_curr)
            @assert(add_edge!(g,N_curr+1,v))
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
            @assert(add_edge!(g,N_curr+1,v))
            add_neighbors_to_array(weights,g,v)
            #add other edges
            for j = 2:m
                v = sample(1:N_curr,Weights(weights[1:N_curr]))
                @assert(sum(weights[N_curr+1:end]) == 0)
                @assert(add_edge!(g,N_curr+1,v))
                @assert(v != N_curr+1)
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
       


function get_connected_subgraph(g,verts)
    gsub = induced_subgraph(g,verts)[1]
    comp = connected_components(gsub)
    comp = sort(comp,by=length)
    return induced_subgraph(gsub,comp[end])[1]
end

function sample_connected_subgraph(g,n_target,tol,n_proposal,max_iters = 10)
    n_curr = 0
    gsub = g
    iters = 0
    while abs(n_curr - n_target) > tol
        verts = sample(1:nv(g),n_proposal,replace=false)
        # verts = mhrw(g,sample(1:nv(g)),n)
        gsub = get_connected_subgraph(g,verts)
        n_curr = nv(gsub)
        iters += 1
        println(n_curr)
        if iters > max_iters
            break
        end
    end
    return gsub
end

function mhrw(g,v,n) #metropolis-hastings random walk
    v_curr = v
    verts = Set{Int64}([v_curr])
    while length(verts) < n
        kv = length(neighbors(g,v_curr))
        w = sample(neighbors(g,v_curr))
        kw = length(neighbors(g,w))
        if rand() <= kv/kw
            v_curr = w
            push!(verts,v_curr)
        end
    end
    collect(verts)
end

function find_closest(arr,N)
    closest = arr[1]
    idx = 1
    for (i,el) in enumerate(arr)
        if abs(el-N) < abs(closest-N)
            closest = el
            idx = i
        end
    end
    return idx,closest
end

function remove_edges(g,k_target)
    N = nv(g)
    n_edges_target = Int(k_target*N/2)
    n_edges_now = ne(g)
    n_to_remove = n_edges_now-n_edges_target
    if n_to_remove <= 0
        return g
    end
    rand_edges = shuffle(collect(edges(g)))
    for i = 1:n_to_remove
        @assert(rem_edge!(g,rand_edges[i]))
    end
    return g
end

function remove_nodes(g,N_target)
    N_curr = nv(g)
    while N_curr > N_target
        N_diff = Int(ceil((N_curr - N_target)/2))
        to_remove = min(N_diff,Int(ceil(N_curr/2)))
        # println(to_remove)
        to_keep = N_curr-to_remove
        nodes_to_keep = sample(1:N_curr,to_keep,replace=false)
        g = induced_subgraph(g,nodes_to_keep)[1]
        # println(nodes_to_remove)
        # removed = rem_vertex!.(g,nodes_to_remove)
        # println(removed)
        @assert(nv(g) == to_keep)
        g = get_largest_connected_component(g)
        N_curr = nv(g)
    end
    return g
end

function get_largest_connected_component(g)
    comp = sort(connected_components(g),by=length)[end]
    return induced_subgraph(g,comp)[1]
end
    


function find_closest_tol(arr,N,tol)
    idx,closest = find_closest(arr,N)
    indices = [idx]
    for (i,el) in enumerate(arr)
        if abs(el-N) < tol*N
            push!(indices,i)
        end
    end
    return indices
end


function sample_subgraph_by_community(g,N_target)
    ls = label_propagation(g)[1]
    sizes = counts(ls)
    idx,closest = find_closest(counts(ls),N_target)
    community = find(x -> x == idx,ls)
    g_out = induced_subgraph(g,community)[1]
    return g_out
end

function sample_subgraph_by_community_given_k(g,N_target,k,tol=0.2)
    ls = label_propagation(g)[1]
    sizes = counts(ls)
    if maximum(sizes) < N_target 
        return copy(g)
    end
    indices = find_closest_tol(counts(ls),N_target,tol)
    println(size(indices))
    gs = []
    ks = []
    for (i,idx) in enumerate(indices)
        community = findall(x -> x == idx,ls)
        g_out = induced_subgraph(g,community)[1]
        push!(ks,mean(degree(g_out)))
        push!(gs,g_out)
    end
    println(ks)
    idx_closest,k_closest = find_closest(ks,k)
    return gs[idx_closest]
end

function print_basic_stats(g)
    N = nv(g)
    k = mean(degree(g))
    C = mean(local_clustering_coefficient(g))
    println("N: $N, k: $k, C:$C")
end


function subsample_from_communities(g,N_target,k_target)
    g_out = sample_subgraph_by_community_given_k(g,N_target,k_target,0.1)
    g_out = remove_nodes(g_out,2*Int(round(N_target/2)))
    g_out = remove_edges(g_out,k_target)
    g_out = get_largest_connected_component(g_out)
    print_basic_stats(g_out)
    return g_out
end





end
