module IndirectRec

using LightGraphs, PyCall, Distributions, GraphConnectivityTheory, Combinatorics

export get_p_known_percolation, get_num_mutual_neighbors,
reverse_vmap, get_mean_p_known_for_node, 
get_components, get_connected_component,
get_binomial_error,get_audience_fraction_binomial_error,
embeddedness,get_p_known_exact,
get_p_known_node_exact_neighborhood_graph,
get_p_known_edge_exact_neighborhood_graph,
sample_graph_edges_multiple_groups,
get_audience_exact,
get_audience_fraction_exact


function graph_from_edges(edges,num_nodes)
    g = Graph(num_nodes)
    for e in edges
        add_edge!(g,e)
    end
    return g
end

    
function reverse_vmap(vmap::Array{Int,1})
    vmap_r = Dict{Int,Int}()
    for (i,v) in enumerate(vmap)
        vmap_r[v] = i
    end
    vmap_r
end

function get_p_known_node_exact_neighborhood_graph(g,v,p)
    return get_p_known_exact(g,v,p)
end

function get_audience_fraction_exact(g,v,w,p)
    all_neighbors = neighbors(g,v)
    k = length(all_neighbors)
    h,vmap = LightGraphs.induced_subgraph(g,all_neighbors)
    vmap = reverse_vmap(vmap)
    vs = vertices(h)
    w = vmap[w]
    return get_p_known_exact(h,w,p)
end

function get_audience_exact(g,v,w,p)
    all_neighbors = neighbors(g,v)
    k = length(all_neighbors)
    return (k-1)*get_audience_fraction_exact(g,v,w,p)
end


function get_p_known_edge_exact_neighborhood_graph(g,v,w,p,q)
    all_neighbors = neighbors(g,v)
    k = length(all_neighbors)
    h,vmap = LightGraphs.induced_subgraph(g,all_neighbors)
    vmap = reverse_vmap(vmap)
    vs = vertices(h)
    w = vmap[w]
    return q/k*(1 + (k-1)*get_p_known_node_exact_neighborhood_graph(h,w,p))
end
    
function get_p_known_exact(g,v,p)
    all_edges = copy(collect(LightGraphs.edges(g)))
    num_edges_total = ne(g)
    num_nodes_total = nv(g)
    println("Checking $(2^num_edges_total) combinations.")
    tot = 0
    for num_edges = 0:num_edges_total
        fac = p^num_edges*(1-p)^(num_edges_total-num_edges)
        for edges_curr in combinations(all_edges,num_edges)
            h = graph_from_edges(edges_curr,num_nodes_total)
            tot += fac*(length(get_connected_component(h,v))-1)
        end
    end
    return tot/(num_nodes_total-1)
end

function get_num_mutual_neighbors(g::LightGraphs.Graph,e::Edge)
    v1 = e.src
    v2 = e.dst
    neighbors1 = LightGraphs.neighbors(g,v1)
    neighbors2 = LightGraphs.neighbors(g,v2)
    num_mutual_neighbors = intersection_length(neighbors1,neighbors2)
    return num_mutual_neighbors
end

function get_mean_p_known_for_node(g::LightGraphs.Graph,v::Int,p_known_fn)
    p_known = 0
    neighbors = LightGraphs.neighbors(g,v)
    for w in neighbors
        p_known += p_known_fn(g,Pair(v,w))
    end
    return p_known/length(neighbors)
end

function get_mean_p_known_on_edge_for_graph(g::LightGraphs.Graph,p_known_fn)
    p_known = 0
    num_edges = 0
    for v in LightGraphs.vertices(g)
        if LightGraphs.degree(g,v) == 0 continue end
        neighbors = LightGraphs.neighbors(g,v)
        for w in neighbors
            p_known += p_known_fn(g,Pair(v,w))
            num_edges += 1
        end
    end
    return p_known/num_edges
end

function get_mean_p_known_on_node_for_graph(g::LightGraphs.Graph,p_known_fn)
    p_known = 0
    num_vertices = 0
    for v in LightGraphs.vertices(g)
        if LightGraphs.degree(g,v) == 0 continue end
        p_known += get_mean_p_known_for_node(g,v,p_known_fn)
        num_vertices += 1
    end
    return p_known/num_vertices
end

###########################Monte Carlo Calculation of p_known##########################3

const GraphComponents = Array{Set{Int}}

function get_components(g::LightGraphs.Graph)
    components::GraphComponents = []
    for v = LightGraphs.vertices(g)
        if !components_contain(components,v)
            new_component = get_connected_component(g,v)
            push!(components,new_component)
        end
    end
    return components
end

function add_node_and_all_neighbors(g::LightGraphs.Graph,v::Int,component::Set{Int})
    push!(component,v)
    for w in LightGraphs.neighbors(g,v)
        if !(w in component)
            add_node_and_all_neighbors(g,w,component)
        end
    end
end
    

function get_connected_component(g::LightGraphs.Graph,v::Int)
    component = Set{Int}()
    add_node_and_all_neighbors(g,v,component)
    return component
end


function components_contain(components::GraphComponents,v::Int)
    for component in components
        if v in component return true end
    end
    return false
end




function get_p_known_percolation(g::LightGraphs.Graph,p::Real,num_trials = 100)
    p_knowns = zeros(size(LightGraphs.vertices(g)))
    singletons = get_singleton_nodes_array(g)
    println(num_trials)
    for i in 1:num_trials
        h = sample_graph_edges(g,p)
        components = get_components(h::LightGraphs.Graph)
        p_knowns += get_p_known_given_components(g,components)
    end
    p_knowns /= num_trials
    return p_knowns,mean(p_knowns[!singletons])
end


#Monte Carlo Calculation of p_known
function get_p_known_given_components(g::LightGraphs.Graph,components::GraphComponents)
    p_knowns = zeros(size(LightGraphs.vertices(g)))
    for v in LightGraphs.vertices(g)
        neighbors = LightGraphs.neighbors(g,v)
        for w in neighbors
            if nodes_in_same_component(v,w,components) p_knowns[v] += 1.0 end
        end
        if length(neighbors) > 0
            p_knowns[v] /= length(neighbors)
        end
    end
    return p_knowns
end

#Monte Carlo Calculation of p_known with maximum path length.
function get_p_known_given_components(g::LightGraphs.Graph,components::GraphComponents,dists::Array{Int,2},max_dist::Int)
    p_knowns = zeros(size(LightGraphs.vertices(g)))
    for v in LightGraphs.vertices(g)
        neighbors = LightGraphs.neighbors(g,v)
        for w in neighbors
            if nodes_in_same_component(v,w,components) && dists[v,w] <= max_dist
                p_knowns[v] += 1.0
            end
        end
        if length(neighbors) > 0
            p_knowns[v] /= length(neighbors)
        end
    end
    return p_knowns
end


function get_singleton_nodes_array(g::LightGraphs.Graph)
    singletons = Array{Bool}(size(LightGraphs.vertices(g)))
    for v in LightGraphs.vertices(g)
        if length(LightGraphs.neighbors(g,v)) == 0
            singletons[v] = true
        else
            singletons[v] = false
        end
    end
    return singletons
end

function nodes_in_same_component(v::Int,w::Int,components::GraphComponents)
    for component in components
        if (v in component) && (w in component)
            return true
        end
    end
    return false
end
            
            
            
function sample_graph_edges(g::LightGraphs.Graph,p::Real)
    h = copy(g)
    edges = copy(collect(LightGraphs.edges(h)))
    for ed in edges
        if rand() < (1-p)
            LightGraphs.rem_edge!(h,ed)
        end
    end
    return h
end

function sample_graph_edges_multiple_groups(g::LightGraphs.Graph,ps::Array{T,1} where T <: Real,edge_groups::Array{Array{Edge,1},1})
    h = copy(g)
    for (i,edge_group) in enumerate(edge_groups)
        p = ps[i]
        for ed in edge_group
            if rand() < (1-p)
                LightGraphs.rem_edge!(h,ed)
            end
        end
    end
    return h
end


function get_p_known_percolation(g::LightGraphs.Graph,p::Real,max_order::Int,num_trials = 100)
    p_knowns = zeros(size(LightGraphs.vertices(g)))
    singletons = get_singleton_nodes_array(g)
    for i in 1:num_trials
        h = sample_graph_edges(g,p)
        components = get_components(h::LightGraphs.Graph)
        # dists = LightGraphs.floyd_warshall_shortest_paths(h).dists
        dists = dijkstra_all_sources_shortest_paths(h) 
        p_knowns += get_p_known_given_components(g,components,dists,max_order)
    end
    p_knowns /= num_trials
    return p_knowns,mean(p_knowns[.!singletons])
end

function dijkstra_all_sources_shortest_paths(g::LightGraphs.Graph)
    dists = zeros(Int,(LightGraphs.nv(g),LightGraphs.nv(g)))
    for v in LightGraphs.vertices(g)
        dists[v,:] = LightGraphs.dijkstra_shortest_paths(g,v).dists
    end
    return dists
end



function get_num_second_neighbors(g::LightGraphs.Graph,e::Pair{Int,Int})
    v1 = e[1]
    v2 = e[2]
    neighbors1 = LightGraphs.neighbors(g,v1)
    neighbors2 = LightGraphs.neighbors(g,v2)
    num_mutual_neighbors = intersection_length(neighbors1,neighbors2)
    return num_mutual_neighbors
end

function get_p_known_percolation(g::LightGraphs.Graph,e::Pair{Int,Int},p::Real,num_trials = 100)
    connecteds = 0
    for i in 1:num_trials
        h = copy(g)
        edges = copy(LightGraphs.edges(h))
        for ed in edges
            if rand() < (1-p)
                LightGraphs.rem_edge!(h,ed)
            end
        end
#         @show LightGraphs.dijkstra_shortest_paths(h,e[1]).parents[e[2]]
        connecteds += LightGraphs.dijkstra_shortest_paths(h,e[1]).parents[e[2]] != 0
    end
    return connecteds/num_trials
end

#this assumes that paths to all neighbors are perfectly dependent (also not entirely true). This overestimates error.
function get_binomial_error(p_est,n_samples)
    return sqrt(p_est*(1-p_est)/n_samples)
end

#binomial error if p is computed as the mean P to reach one of k neighbors. This 
#assumes that paths to all neighbors are perfectly independent (not true). This underestimates error.
function get_audience_fraction_binomial_error(p_est,n_samples,k)
    return sqrt(p_est*(1-p_est)/(n_samples*k))
end

function embeddedness(g)
    (degree(g)-1).*local_clustering_coefficient(g)
end



end