module CliquePercolation 

using IndirectRec, GraphConnectivityTheory, LightGraphs, PyCall, Distributions

export get_p_known_clique_percolation, get_p_known_clique_theory, get_p_known_clique_neighbor_to_neighbor_theory,
produce_clique_graph,
get_p_known_from_neighbor_to_other_neighbor


function get_p_known_clique_percolation(g::LightGraphs.Graph,p::Real,num_trials = 100)
    p_knowns = zeros(size(LightGraphs.vertices(g)))
    singletons = IndirectRec.get_singleton_nodes_array(g)
    for i in 1:num_trials
        h = IndirectRec.sample_graph_edges(g,p)
        p_knowns += get_p_known_clique(g,h)
    end
    p_knowns /= num_trials
    return p_knowns,mean(p_knowns[!singletons])
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
    vs_new = zeros(vs)
    for i in 1:length(vs)
        vs_new[i] = vmap[vs[i]]
    end
    vs_new
end
    
function reverse_vmap(vmap::Array{Int,1})
    vmap_r = Dict{Int,Int}()
    for (i,v) in enumerate(vmap)
        vmap_r[v] = i
    end
    vmap_r
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
    curr = 0
    p_c = p*c
    An_vec = GraphConnectivityTheory.memoize_An(BigInt(N),BigFloat(p_c))
    for i = 1:N
        curr += An_vec[i]*binomial(BigInt(N),BigInt(i))*(1-p_c)^(i*(N-i))*(i/N)*(1-p)^(i)
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

function get_p_known_clique_neighbor_to_neighbor_theory(k::Int,c::Float64,n::Int,p::Float64)
    #n = get_num_mutual_neighbors(g,Pair(v,w))
    #k = degree(g,v)
    if k == 1
        return 0
    elseif k == 2
        if n > 0
            return p
        else
            return 0
        end
    end
        
    c1 = (c*(k*(k-1))/2-n)/((k-1)*(k-2)/2)
    return Float64(get_p_known_clique_theory(k-1,c1*(k-1)/n,n/(k-1)*p))
end

function get_p_known_clique_neighbor_to_neighbor_reliability_theory(k::Int,c::Float64,p::Float64)
    #n = get_num_mutual_neighbors(g,Pair(v,w))
    #k = degree(g,v)
    if k == 1
        return 0
    elseif k == 2
        if n > 0
            return p
        else
            return 0
        end
    end

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
    vs_without_w = deleteat!(collect(vs), findin(vs, w))
    for i in 1:num_trials
        h = IndirectRec.sample_graph_edges(clique,p)
        components = IndirectRec.get_components(h)
        p_known += get_p_known_given_components(components,w,vs_without_w)
    end
    return p_known/num_trials
end

end
