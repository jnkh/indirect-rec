module CliquePercolation 

using IndirectRec, GraphConnectivityTheory, LightGraphs, PyCall, Distributions

export get_p_known_clique_percolation,
get_p_known_clique_theory,
get_p_known_clique_neighbor_to_neighbor_theory,
get_p_known_clique_neighbor_to_neighbor_reliability_theory,
produce_clique_graph,
get_p_known_from_neighbor_to_other_neighbor,
get_p_known_from_neighbor_to_other_neighbor_given_first,
get_p_known_from_all_neighbors_to_other_neighbor,
edgewise_critical_b_c,
get_node_critical_thresh_theory,
get_edge_critical_thresh_theory,
get_p_known_from_neighbor_to_other_neighbor_global_given_first,
get_audience_fraction,
get_audience,
get_audience_fraction_theory,
get_audience_theory,
get_audience_fraction_global,
get_audience_global,
get_central_audience_fraction_theory,
get_central_audience_theory,
get_central_audience_fraction,
get_central_audience


#critical b-c ratio as seen by a given node of degree k considering whether to defect with one of its neighbors
#if this is larger than the actual b/c ratio, then the node defects.
function edgewise_critical_b_c(rho_prime,k,P_neighbor_known)
    if k == 1
        return rho_prime/(rho_prime-1)
    elseif k == 0
        return 0
    elseif k > 1
        return (rho_prime*(k-1))/(P_neighbor_known*(k-1)+1)
    end
end



function get_p_known_clique_percolation(g::LightGraphs.Graph,p::Real,num_trials = 100)
    p_knowns = zeros(size(LightGraphs.vertices(g)))
    singletons = IndirectRec.get_singleton_nodes_array(g)
    for i in 1:num_trials
        h = IndirectRec.sample_graph_edges(g,p)
        p_knowns += get_p_known_clique(g,h)
    end
    p_knowns /= num_trials
    return p_knowns,mean(p_knowns[.!singletons])
end


function get_p_known_clique(g::LightGraphs.Graph,h::LightGraphs.Graph)
    p_knowns = zeros(size(LightGraphs.vertices(g)))
    for v in LightGraphs.vertices(g)
        p_knowns[v] = get_p_known_clique(g,h,v)
    end
    return p_knowns
end

function get_p_known_clique(g::LightGraphs.Graph,h::LightGraphs.Graph,v::Int)
    curr_neighbors = LightGraphs.neighbors(g,v)
    clique_members = vcat([v],curr_neighbors) 
    clique,vmap = LightGraphs.induced_subgraph(h,clique_members)
    vmap = reverse_vmap(vmap)
    curr_neighbors_mapped = map_vertices(vmap,curr_neighbors)
    components = IndirectRec.get_components(clique)
    p_known = get_p_known_given_components(components,vmap[v],curr_neighbors_mapped)
    return p_known
end

function map_vertices(vmap::Dict{Int,Int},vs::Array{Int,1})
    vs_new = 0*similar(vs)
    for i in 1:length(vs)
        vs_new[i] = vmap[vs[i]]
    end
    vs_new
end

    
function get_p_known_given_components(components::IndirectRec.GraphComponents,v::Int,candidates::Array{Int,1})
    p_known = 0
    nc = length(candidates)
    if nc > 0
        for w in candidates
            if IndirectRec.nodes_in_same_component(v,w,components)
                p_known += 1.0
            end
        end
        p_known /= nc
    end
    return p_known
end

function get_p_known_all_pairs_given_components(components::IndirectRec.GraphComponents,candidates::Array{Int,1})
    p_known = 0
    counts = 0
    nc = length(candidates)
    if nc > 0
        for v in candidates
            for w in candidates
                if v != w
                    if IndirectRec.nodes_in_same_component(v,w,components)
                        p_known += 1.0
                    end
                    counts += 1.0
                end
            end
        end
    end
    if counts > 0
        return p_known/counts
    else
        return 0
    end
end
    
    


function produce_clique_graph(N,c::Float64)
    h = erdos_renyi(N,c)
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

function get_p_known_clique_theory(k::Int,c::Float64,p::Float64)
    if k == 0
        return 0
    end
    N = k 
    curr = BigFloat(0.0)
    p_c = p*c
    An_vec = GraphConnectivityTheory.memoize_An(BigInt(N),BigFloat(p_c))
    for i = 1:N
        curr += An_vec[i]*binomial(BigInt(N),BigInt(i))*(1-p_c)^(i*(N-i))*(i/N)*(1-p)^(i)
    end
    return 1 - curr
end   

function get_central_audience_theory(g::LightGraphs.Graph,p1::Float64,p2::Float64,v::Int)
    k = degree(g,v)
    c = local_clustering_coefficient(g,v)
    return get_central_audience_theory(k,c,p1,p2)
end

function get_central_audience_theory(k::Int,c::Float64,p1::Float64,p2::Float64)
    return k*Float64(get_central_audience_fraction_theory(k,c,p1,p2))
end


function get_central_audience_fraction_theory(k::Int,c::Float64,p1::Float64,p2::Float64)
    if k == 0
        return 0
    end
    N = k 
    curr = BigFloat(0.0)
    p_c = p2*c
    An_vec = GraphConnectivityTheory.memoize_An(BigInt(N),BigFloat(p_c))
    for i = 1:N
        curr += An_vec[i]*binomial(BigInt(N),BigInt(i))*(1-p_c)^(i*(N-i))*(i/N)*(1-p1)^(i)
    end
    return 1 - curr
end

function get_p_known_clique_theory_approx(k::Int,c::Float64,p::Float64)
    N = k 
    curr = 0
    p_c = p*c
    An_vec = get_An_approx(collect(1:N),p_c)
    for i = 1:N
        curr += An_vec[i]*binomial(BigInt(N),BigInt(i))*(1-p_c)^(i*(N-i))*(i/N)*(1-p)^(i)
    end
    return 1 - curr
end 

function get_audience_fraction_theory(k::Int,c::Float64,n::Int,p::Float64)
    P_other_neighbors = 0.0
    if n > 0
        c1 = (c*(k*(k-1))/2-n)/((k-1)*(k-2)/2)
        P_other_neighbors = Float64(get_p_known_clique_theory(k-1,c1*(k-1)/n,n/(k-1)*p))
    end
    return P_other_neighbors
end

function get_audience_theory(k::Int,c::Float64,n::Int,p::Float64)
    P_other_neighbors = get_audience_fraction_theory(k,c,n,p)
    ret = (k-1)*P_other_neighbors
    return ret
end

function get_audience_theory(g::LightGraphs.Graph,p::Real,v::Int,w::Int)
    k = degree(g,v)
    c = local_clustering_coefficient(g,v)
    n = get_num_mutual_neighbors(g,Edge(v,w))
    return get_audience_theory(k,c,n,p)
end


function get_p_known_clique_neighbor_to_neighbor_theory(k::Int,c::Float64,n::Int,p::Float64)
    #n = get_num_mutual_neighbors(g,Pair(v,w))
    #k = degree(g,v)
    P_other_neighbors = 0.0

    # if n == 0
    #     return 0.0
    # end

    # if k == 1
    #     return 0
    # elseif k == 2
    #     if n > 0
    #         return p
    #     else
    #         return 0.0
    #     end
    # end
        
    if n > 0
        c1 = (c*(k*(k-1))/2-n)/((k-1)*(k-2)/2)
        P_other_neighbors = Float64(get_p_known_clique_theory(k-1,c1*(k-1)/n,n/(k-1)*p))
    end

    ret = p/k*(1 + (k-1)*P_other_neighbors)
    return ret
end

function get_p_known_clique_neighbor_to_neighbor_reliability_theory(k::Int,c::Float64,p::Float64)
    #n = get_num_mutual_neighbors(g,Pair(v,w))
    #k = degree(g,v)
    if k == 1
        return 0
    end
    # elseif k == 2
    #     if n > 0
    #         return p
    #     else
    #         return 0
    #     end
    # end

    return Float64(GraphConnectivityTheory.get_Tn_memoized(BigInt(k),BigFloat(p*c)))
end

#### TODO TEST
####This quantity is not symmetric!
function get_p_known_from_neighbor_to_other_neighbor(g::LightGraphs.Graph,p::Real,v::Int,w::Int,num_trials = 100)
    p_known = 0
    curr_neighbors = LightGraphs.neighbors(g,v)
    clique,vmap = LightGraphs.induced_subgraph(g,curr_neighbors)
    vmap = reverse_vmap(vmap)
    vs = vertices(clique)
    w = vmap[w]

    add_vertex!(clique)
    source_vertex = nv(clique)
    add_edge!(clique,Edge(source_vertex,w)) #add back single vertex

    # vs_without_w = deleteat!(collect(vs), findin(vs, w))
    for i in 1:num_trials
        h = IndirectRec.sample_graph_edges(clique,p)
        components = IndirectRec.get_components(h)
        p_known += get_p_known_given_components(components,source_vertex,collect(vs))
    end
    return p_known/num_trials
end


function get_audience_fraction(g::LightGraphs.Graph,p::Real,v::Int,w::Int,num_trials = 100)
    frac = 0
    curr_neighbors = LightGraphs.neighbors(g,v)
    clique,vmap = LightGraphs.induced_subgraph(g,curr_neighbors)
    vmap = reverse_vmap(vmap)
    vs = vertices(clique)
    w = vmap[w]
    source_vertex = w

    vs_without_w = deleteat!(collect(vs), findfirst(x -> x == w,vs))
    for i in 1:num_trials
        h = IndirectRec.sample_graph_edges(clique,p)
        components = IndirectRec.get_components(h)
        frac += get_p_known_given_components(components,source_vertex,collect(vs_without_w))
    end
    return frac/num_trials
end

function get_audience(g::LightGraphs.Graph,p::Real,v::Int,w::Int,num_trials = 100)
    k = length(LightGraphs.neighbors(g,v))
    return (k-1)*get_audience_fraction(g,p,v,w,num_trials)
end

function get_audience_fraction_global(g_orig::LightGraphs.Graph,p1::Real,p2::Real,v::Int,w::Int,num_trials = 100)
    frac = 0
    g = copy(g_orig)
    curr_neighbors = collect(LightGraphs.neighbors(g,v))
    source_vertex = w
#     println(source_vertex)
#     println(curr_neighbors)
    neighborhood = Set(curr_neighbors)
#     rem_vertex!(h,v) #remove all edges that go via the actor
    neighbor_edges = Edge[]
    global_edges = Edge[]
    for ed in collect(edges(g))
        v1 = ed.src
        v2 = ed.dst
        if v1 == v || v2 == v
            rem_edge!(g,ed)
        elseif v1 in neighborhood && v2 in neighborhood
            push!(neighbor_edges,ed)
        else
            push!(global_edges,ed)
        end
    end
    edge_groups = [neighbor_edges,global_edges]
    ps = [p1,p2]
#     println(curr_neighbors)

    neighbors_without_source = deleteat!(collect(curr_neighbors), findfirst(x -> x == source_vertex,curr_neighbors))
    for i in 1:num_trials
        gtemp = IndirectRec.sample_graph_edges_multiple_groups(g,ps,edge_groups)
        components = IndirectRec.get_components(gtemp)
#         println(curr_neighbors)
        frac += get_p_known_given_components(components,source_vertex,neighbors_without_source)
    end
    return frac/num_trials
end


function get_audience_global(g_orig::LightGraphs.Graph,p1::Real,p2::Real,v::Int,w::Int,num_trials = 100)
    k = length(LightGraphs.neighbors(g_orig,v))
    return (k-1)*get_audience_fraction_global(g_orig,p1,p2,v,w,num_trials)
end


function get_audiences(g,p,theory=false,ntrials=100)
    edges = []
    audiences = []
    for v in vertices(g)
        for w in neighbors(g,v)
            push!(edges,(v,w))
            if theory
                push!(audiences,get_audience_theory(g,p,v,w))
            else
                push!(audiences,get_audience(g,p,v,w,ntrials))
            end
        end
    end
    return edges,audiences
end

function get_audiences_by_node(g,p,theory=false,ntrials=100)
    audiences = []
    total = 0.0
    for v in vertices(g)
        nij = 0.0
        neighs = neighbors(g,v)
        for w in neighs 
            if theory
                val = get_audience_theory(g,p,v,w)
            else
                val = get_audience(g,p,v,w,ntrials)
            end
            nij += val
            total += val
        end
        nij /= length(neighs)
        push!(audiences,nij)
    end
    nij_mean = total/(2*length(edges(g)))
    return audiences,nij_mean
end

function get_audiences_by_recipient_node(g,p,theory=false,ntrials=100)
    audiences = []
    total = 0.0
    for v in vertices(g)
        nij = 0.0
        neighs = neighbors(g,v)
        for w in neighs 
            if theory
                val = get_audience_theory(g,p,w,v)
            else
                val = get_audience(g,p,w,v,ntrials)
            end
            nij += val
            total += val
        end
        nij /= length(neighs)
        push!(audiences,nij)
    end
    nij_mean = total/(2*length(edges(g)))
    return audiences,nij_mean
end

function get_central_audience(g_orig::LightGraphs.Graph,p1::Real,p2::Real,v::Int,num_trials = 100)
    k = degree(g_orig,v)
    return k* get_central_audience_fraction(g_orig,p1,p2,v,num_trials)
end

function get_central_audience_fraction(g_orig::LightGraphs.Graph,p1::Real,p2::Real,v::Int,num_trials = 100)
    curr_neighbors = LightGraphs.neighbors(g_orig,v)
    clique_members = vcat([v],curr_neighbors) 
    clique_orig,vmap = LightGraphs.induced_subgraph(g_orig,clique_members)
    vmap = reverse_vmap(vmap)
    curr_neighbors_mapped = map_vertices(vmap,curr_neighbors)
    v_mapped = vmap[v]
    observation_edges = Edge[]
    neighborhood_edges = Edge[]
    for ed in collect(edges(clique_orig))
        v1 = ed.src
        v2 = ed.dst
        if v1 == v_mapped || v2 == v_mapped
            push!(observation_edges,ed)
        else
            push!(neighborhood_edges,ed)
        end
    end
    edge_groups = [observation_edges,neighborhood_edges]
    ps = [p1,p2]

    audience_frac = 0.0
    for i in 1:num_trials
        clique_tmp = IndirectRec.sample_graph_edges_multiple_groups(clique_orig,ps,edge_groups)
        components = IndirectRec.get_components(clique_tmp)
        audience_frac += get_p_known_given_components(components,vmap[v],curr_neighbors_mapped)
    end
    return audience_frac/num_trials
end


function get_p_known_from_neighbor_to_other_neighbor_given_first(g::LightGraphs.Graph,p::Real,v::Int,w::Int,num_trials = 100)
    p_known = 0
    curr_neighbors = LightGraphs.neighbors(g,v)
    clique,vmap = LightGraphs.induced_subgraph(g,curr_neighbors)
    vmap = reverse_vmap(vmap)
    vs = vertices(clique)
    w = vmap[w]

    # add_vertex!(clique)
    # source_vertex = nv(clique)
    # add_edge!(clique,Edge(source_vertex,w)) #add back single vertex
    source_vertex = w

    # vs_without_w = deleteat!(collect(vs), findin(vs, w))
    for i in 1:num_trials
        h = IndirectRec.sample_graph_edges(clique,p)
        components = IndirectRec.get_components(h)
        p_known += get_p_known_given_components(components,source_vertex,collect(vs))
    end
    return p_known/num_trials
end

function get_p_known_from_neighbor_to_other_neighbor_global_given_first(g_orig::LightGraphs.Graph,p1::Real,p2::Real,v::Int,w::Int,num_trials = 100)
    p_known = 0
    g = copy(g_orig)
    curr_neighbors = collect(LightGraphs.neighbors(g,v))
    source_vertex = w
#     println(source_vertex)
#     println(curr_neighbors)
    neighborhood = Set(curr_neighbors)
#     rem_vertex!(h,v) #remove all edges that go via the actor
    neighbor_edges = Edge[]
    global_edges = Edge[]
    for ed in collect(edges(g))
        v1 = ed.src
        v2 = ed.dst
        if v1 == v || v2 == v
            rem_edge!(g,ed)
        elseif v1 in neighborhood && v2 in neighborhood
            push!(neighbor_edges,ed)
        else
            push!(global_edges,ed)
        end
    end
    edge_groups = [neighbor_edges,global_edges]
    ps = [p1,p2]
#     println(curr_neighbors)

    for i in 1:num_trials
        gtemp = IndirectRec.sample_graph_edges_multiple_groups(g,ps,edge_groups)
        components = IndirectRec.get_components(gtemp)
#         println(curr_neighbors)
        p_known += get_p_known_given_components(components,source_vertex,curr_neighbors)
    end
    return p_known/num_trials
end


function get_p_known_from_all_neighbors_to_other_neighbor(g::LightGraphs.Graph,p::Real,v::Int,num_trials = 100)
    p_known = 0
    all_neighbors = LightGraphs.neighbors(g,v)
    clique,vmap = LightGraphs.induced_subgraph(g,all_neighbors)
    vmap = reverse_vmap(vmap)
    vs = collect(vertices(clique))
    # vs_without_w = deleteat!(collect(vs), findin(vs, w))
    for i in 1:num_trials
        h = IndirectRec.sample_graph_edges(clique,p)
        components = IndirectRec.get_components(h)
        p_known += get_p_known_all_pairs_given_components(components,vs)
    end
    return p_known/num_trials
end



function get_edge_critical_thresh_theory(g,v,w,p)
    k = degree(g,v)
    c = local_clustering_coefficient(g,v)
    n = get_num_mutual_neighbors(g,Edge(v,w))
    P_tilde_th = get_p_known_clique_neighbor_to_neighbor_theory(k,c,n,p)
    ret = P_tilde_th
    return k*ret
end

function get_node_critical_thresh_theory(g,v,p)
    k = degree(g,v)
    c = local_clustering_coefficient(g,v)
    ret = Float64(get_p_known_clique_theory(k,c,p))
    return ret
end

end

