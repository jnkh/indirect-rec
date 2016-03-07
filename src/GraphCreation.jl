module GraphCreation

using PyCall, LightGraphs

export create_graph, networkx_to_lightgraph

function create_graph(N,k,graph_type=:erdos_renyi)
    p_edge = k/(N-1)
    if graph_type == :erdos_renyi
        g = LightGraphs.erdos_renyi(N,p_edge)
    elseif graph_type == :watts_strogatz
        g = LightGraphs.watts_strogatz(N,Int(round((N-1)*p_edge)),0.5)
    elseif graph_type == :random_regular
        g = LightGraphs.random_regular_graph(N,Int(round((N-1)*p_edge)))
    elseif graph_type == :powerlaw_cluster
        g = powerlaw_cluster_graph(N,Int(round(0.5*(N-1)*p_edge)),1.0)
    elseif graph_type == :fb
        g = read_edgelist("../data/facebook_combined.txt")
    else
        error("invalid graph type")
    end
    g
end

@pyimport networkx as nx
function networkx_to_lightgraph(G)
    g = LightGraphs.Graph(length(nx.nodes(G)))
    for e in nx.edges(G)
        LightGraphs.add_edge!(g,e[1]+1,e[2]+1)
    end
    g
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
    G = nx.read_edgelist(filename,nodetype=int)
    return networkx_to_lightgraph(G)
end

end