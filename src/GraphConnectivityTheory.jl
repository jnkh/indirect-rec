module GraphConnectivityTheory 

using LightGraphs, PyCall, Distributions

export get_p_known_percolation_theory, percolation_erdos_renyi,
 intersection_length, choose, get_Tn_memoized,
 get_p_known_first_order

function get_p_known_first_order(g::LightGraphs.Graph,e::Pair{Int,Int},p::Real)
    return get_p_known_first_order(get_num_mutual_neighbors(g,e),p)
end

function get_p_known_first_order_theory(g::LightGraphs.Graph,p::Real)
    return get_p_known_first_order(get_num_mutual_neighbors_theory(g),p)
end


function get_num_mutual_neighbors_theory(g::LightGraphs.Graph)
    num_vertices= LightGraphs.nv(g)
    p_edge = 2*LightGraphs.ne(g)/(num_vertices*(num_vertices-1))
    num_mutual_neighbors = (num_vertices - 2)*p_edge^2
    return num_mutual_neighbors
end
    
function get_p_known_first_order(num_mutual_neighbors::Real,p::Real)
    return 1 - (1-p)*(1-p^2)^num_mutual_neighbors
end

function get_p_known_zeroth_order(g::LightGraphs.Graph,e::Pair{Int,Int},p::Real)
    return p 
end



#########################Calculations for p_known from number of paths, higher order (>2)#####################

function get_p_known_second_order(g::LightGraphs.Graph,e::Pair{Int,Int},p::Real)
    return get_p_known_second_order(get_num_mutual_neighbors(g,e),get_num_second_neighbors(g,e),p)
end

function get_p_known_second_order_theory(g::LightGraphs.Graph,p::Real)
    return get_p_known_second_order(get_num_mutual_neighbors_theory(g),get_num_second_neighbors_theory(g),p)
end
function get_num_second_neighbors_theory(g::LightGraphs.Graph)
    num_vertices= LightGraphs.nv(g)
    p_edge = 2*LightGraphs.ne(g)/(num_vertices*(num_vertices-1))
    num_second_neighbors = (num_vertices - 2)*(num_vertices - 3)*p_edge^3
    return num_second_neighbors
end
    
function get_p_known_second_order(num_mutual_neighbors::Real,num_second_neighbors::Real,p::Real)
    return 1 - (1-p)*(1-p^2)^num_mutual_neighbors*(1-p^3)^num_second_neighbors
end

function get_nth_neighbors_theory(g::LightGraphs.Graph,n::Int)
    num_vertices= LightGraphs.nv(g)
    p_edge = 2*LightGraphs.ne(g)/(num_vertices*(num_vertices-1))
    num = 1
    for i = 1:n
        num *= (num_vertices - 1 - i)
    end
    num *= p_edge^(n + 1)
    return num
end

function get_p_known_nth_order_theory(g::LightGraphs.Graph,p::Real,n::Int)
    prod = 1 - p
    for i = 1:n
        prod *= (1-p^(i+1))^get_nth_neighbors_theory(g,i)
    end
    return 1- prod
end


###########################  Percolation Theory #########################
###########################Reliability function calculation of p_known#########################

function memoize_An(n::BigInt,p::BigFloat)
    Ans = Array{BigFloat}(n)
    Ans[1] = 1
    for n_curr = 2:n
        term::BigFloat = 0
        for j = 1:n_curr-1
            term += binomial(BigInt(n_curr-1),BigInt(j-1)) * Ans[j] * (1-p)^(j*(n_curr-j))
        end
        Ans[n_curr] = 1-term
    end
    return Ans
end    

# function get_An(n,p)
#     if n == 1 return 1 end
#     term = 0
#     for j = 1:n-1
#         term += binomial(BigInt(n-1),BigInt(j-1)) * get_An(j,p) * (1-p)^(j*(n-j))
#     end
#     return 1-term
# end

function get_Tn_memoized(n::BigInt,p::BigFloat)
    Ans::Array{BigFloat,1} = memoize_An(n,p)
    term::BigFloat = 0
    for j = 2:n
        term += binomial(BigInt(n-2),BigInt(j-2)) * Ans[j] * (1-p)^(j*(n-j))
    end
    return term
end

# function get_Tn(n,p)
#     term = 0
#     for j = 2:n
#         term += binomial(BigInt(n-2),BigInt(j-2)) * get_An(j,p) * (1-p)^(j*(n-j))
#     end
#     return term
# end


function get_p_known_percolation_theory(g::LightGraphs.Graph,p)
    n = LightGraphs.nv(g)
    p_edge = 2*LightGraphs.ne(g)/(n*(n-1))
    return get_p_known_percolation_theory(n,p_edge,p)
end

function get_p_known_percolation_theory(n,p_edge,p)
    reliability = get_Tn_memoized(BigInt(n),BigFloat(p*p_edge))
    if p == p_edge == 1.0 return 1.0 end
    return p + (1-p)*(reliability - p_edge*p)/(1-p_edge*p)
end

function percolation_erdos_renyi(n::Int,p::Float64)
    return convert(Float64,get_Tn_memoized(BigInt(n),BigFloat(p)))
end


#############  Utilities  ###############

function intersection_length(a::Array{Int,1},b::Array{Int,1})
    l = 0
    b = sort(b)
    for v in a
        l += length(searchsorted(b,v))
    end
    l
end

function choose(n::Int,k::Int)
    return factorial(n)/(factorial(n-k)*factorial(k))
end



end