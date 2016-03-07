module IndirectRec

using LightGraphs, PyCall, Distributions

export get_p_known_percolation


function get_num_mutual_neighbors(g::LightGraphs.Graph,e::Pair{Int,Int})
    v1 = e[1]
    v2 = e[2]
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

typealias GraphComponents Array{Set{Int}}

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
    singletons = Array(Bool,size(LightGraphs.vertices(g)))
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
    return p_knowns,mean(p_knowns[!singletons])
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




end