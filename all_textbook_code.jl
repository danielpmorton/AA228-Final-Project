using LightGraphs
using Distributions
using JuMP
using GLPK

## BEGIN ALGORITHM 2.1 ###########################
struct Variable
    name::Symbol
    m::Int # number of possible values
end

const Assignment = Dict{Symbol,Int}
const FactorTable = Dict{Assignment,Float64}

struct Factor
    vars::Vector{Variable}
    table::FactorTable
end

variablenames(ϕ::Factor) = [var.name for var in ϕ.vars]

select(a::Assignment, varnames::Vector{Symbol}) = Assignment(n=>a[n] for n in varnames)

function assignments(vars::AbstractVector{Variable})
    names = [var.name for var in vars]
    return vec([Assignment(n=>v for (n,v) in zip(names, values)) for values in product((1:v.m for v in vars)...)])
end

function normalize!(ϕ::Factor)
    z = sum(p for (a,p) in ϕ.table)
    for (a,p) in ϕ.table
        ϕ.table[a] = p/z
    end
    return ϕ
end
## END ALGORITHM 2.1 ############################

## BEGIN ALGORITHM 2.2 ###########################
struct BayesianNetwork
    vars::Vector{Variable}
    factors::Vector{Factor}
    graph::SimpleDiGraph{Int64}
end
## END ALGORITHM 2.2 ############################

## BEGIN ALGORITHM 3.1 ###########################
function Base.:*(ϕ::Factor, ψ::Factor)
    ϕnames = variablenames(ϕ)
    ψnames = variablenames(ψ)
    ψonly = setdiff(ψ.vars, ϕ.vars)
    table = FactorTable()
    for (ϕa,ϕp) in ϕ.table
        for a in assignments(ψonly)
            a = merge(ϕa, a)
            ψa = select(a, ψnames)
            table[a] = ϕp * get(ψ.table, ψa, 0.0)
        end
    end
    vars = vcat(ϕ.vars, ψonly)
    return Factor(vars, table)
end
## END ALGORITHM 3.1 #########################

## BEGIN ALGORITHM 3.2 ###########################
function marginalize(ϕ::Factor, name)
    table = FactorTable()
    for (a, p) in ϕ.table
        a′ = delete!(copy(a), name)
        table[a′] = get(table, a′, 0.0) + p
    end
    vars = filter(v -> v.name != name, ϕ.vars)
    return Factor(vars, table)
end
## END ALGORITHM 3.2 #############################

## BEGIN ALGORITHM 3.3 ###########################
in_scope(name, ϕ) = any(name == v.name for v in ϕ.vars)

function condition(ϕ::Factor, name, value)
    if !in_scope(name, ϕ)
        return ϕ
    end
    table = FactorTable()
    for (a, p) in ϕ.table
        if a[name] == value
            table[delete!(copy(a), name)] = p
        end
    end
    vars = filter(v -> v.name != name, ϕ.vars)
    return Factor(vars, table)
end

function condition(ϕ::Factor, evidence)
    for (name, value) in pairs(evidence)
        ϕ = condition(ϕ, name, value)
    end
    return ϕ
end
## END ALGORITHM 3.3 #############################

## BEGIN ALGORITHM 3.4 ###########################
struct ExactInference end

function infer(M::ExactInference, bn, query, evidence)
    ϕ = prod(bn.factors)
    ϕ = condition(ϕ, evidence)
    for name in setdiff(variablenames(ϕ), query)
        ϕ = marginalize(ϕ, name)
    end
    return normalize!(ϕ)
end
## END ALGORITHM 3.4 #############################

## BEGIN ALGORITHM 3.5 ###########################
struct VariableElimination
    ordering # array of variable indices
end

function infer(M::VariableElimination, bn, query, evidence)
    Φ = [condition(ϕ, evidence) for ϕ in bn.factors]
    for i in M.ordering
        name = bn.vars[i].name
        if name ∉ query
            inds = findall(ϕ->in_scope(name, ϕ), Φ)
            if !isempty(inds)
                ϕ = prod(Φ[inds])
                deleteat!(Φ, inds)
                ϕ = marginalize(ϕ, name)
                push!(Φ, ϕ)
            end
        end
    end
    return normalize!(prod(Φ))
end
## END ALGORITHM 3.5 #############################

## BEGIN ALGORITHM 3.6 ###########################
function Base.rand(ϕ::Factor)
    tot, p, w = 0.0, rand(), sum(values(ϕ.table))
    for (a,v) in ϕ.table
        tot += v/w
        if tot >= p
            return a
        end
    end
    return Assignment()
end

function Base.rand(bn::BayesianNetwork)
    a = Assignment()
    for i in topological_sort(bn.graph)
        name, ϕ = bn.vars[i].name, bn.factors[i]
        a[name] = rand(condition(ϕ, a))[name]
    end
    return a
end
## END ALGORITHM 3.6 #############################

## BEGIN ALGORITHM 3.7 ###########################
struct DirectSampling
    m # number of samples
end

function infer(M::DirectSampling, bn, query, evidence)
    table = FactorTable()
    for i in 1:(M.m)
        a = rand(bn)
        if all(a[k] == v for (k,v) in pairs(evidence))
            b = select(a, query)
            table[b] = get(table, b, 0) + 1
        end
    end
    vars = filter(v->v.name ∈ query, bn.vars)
    return normalize!(Factor(vars, table))
end
## END ALGORITHM 3.7 #############################

## BEGIN ALGORITHM 3.8 ###########################
struct LikelihoodWeightedSampling
    m # number of samples
end

function infer(M::LikelihoodWeightedSampling, bn, query, evidence)
    table = FactorTable()
    ordering = topological_sort(bn.graph)
    for i in 1:(M.m)
        a, w = Assignment(), 1.0
        for j in ordering
            name, ϕ = bn.vars[j].name, bn.factors[j]
            if haskey(evidence, name)
                a[name] = evidence[name]
                w *= ϕ.table[select(a, variablenames(ϕ))]
            else
                a[name] = rand(condition(ϕ, a))[name]
            end
        end
        b = select(a, query)
        table[b] = get(table, b, 0) + w
    end
    vars = filter(v->v.name ∈ query, bn.vars)
    return normalize!(Factor(vars, table))
end
## END ALGORITHM 3.8 #############################

## BEGIN ALGORITHM 3.9 ###########################
function blanket(bn, a, i)
    name = bn.vars[i].name
    val = a[name]
    a = delete!(copy(a), name)
    Φ = filter(ϕ -> in_scope(name, ϕ), bn.factors)
    ϕ = prod(condition(ϕ, a) for ϕ in Φ)
    return normalize!(ϕ)
end
## END ALGORITHM 3.9 #############################

## BEGIN ALGORITHM 3.10 ###########################
function update_gibbs_sample!(a, bn, evidence, ordering)
    for i in ordering
        name = bn.vars[i].name
        if !haskey(evidence, name)
            b = blanket(bn, a, i)
            a[name] = rand(b)[name]
        end
    end
end

function gibbs_sample!(a, bn, evidence, ordering, m)
    for j in 1:m
        update_gibbs_sample!(a, bn, evidence, ordering)
    end
end

struct GibbsSampling
    m_samples # number of samples to use
    m_burnin # number of samples to discard during burn-in
    m_skip # number of samples to skip for thinning
    ordering # array of variable indices
end

function infer(M::GibbsSampling, bn, query, evidence)
    table = FactorTable()
    a = merge(rand(bn), evidence)
    gibbs_sample!(a, bn, evidence, M.ordering, M.m_burnin)
    for i in 1:(M.m_samples)
        gibbs_sample!(a, bn, evidence, M.ordering, M.m_skip)
        b = select(a, query)
        table[b] = get(table, b, 0) + 1
    end
    vars = filter(v->v.name ∈ query, bn.vars)
    return normalize!(Factor(vars, table))
end
## END ALGORITHM 3.10 #############################

## BEGIN ALGORITHM 3.11 ###########################
function infer(D::MvNormal, query, evidencevars, evidence)
    μ, Σ = D.μ, D.Σ.mat
    b, μa, μb = evidence, μ[query], μ[evidencevars]
    A = Σ[query,query]
    B = Σ[evidencevars,evidencevars]
    C = Σ[query,evidencevars]
    μ = μ[query] + C * (B\(b - μb))
    Σ = A - C * (B \ C')
    return MvNormal(μ, Σ)
end
## END ALGORITHM 3.11 #############################

## BEGIN ALGORITHM 4.1 ###########################
function sub2ind(siz, x)
    k = vcat(1, cumprod(siz[1:end-1]))
    return dot(k, x .- 1) + 1
end

function statistics(vars, G, D::Matrix{Int})
    n = size(D, 1)
    r = [vars[i].m for i in 1:n]
    q = [prod([r[j] for j in inneighbors(G,i)]) for i in 1:n]
    M = [zeros(q[i], r[i]) for i in 1:n]
    for o in eachcol(D)
        for i in 1:n
            k = o[i]
            parents = inneighbors(G,i)
            j = 1
            if !isempty(parents)
                j = sub2ind(r[parents], o[parents])
            end
            M[i][j,k] += 1.0
        end
    end
    return M
end
## END ALGORITHM 4.1 #############################

## BEGIN ALGORITHM 4.2 ###########################
function prior(vars, G)
    n = length(vars)
    r = [vars[i].m for i in 1:n]
    q = [prod([r[j] for j in inneighbors(G,i)]) for i in 1:n]
    return [ones(q[i], r[i]) for i in 1:n]
end
## END ALGORITHM 4.2 #############################

## BEGIN ALGORITHM 4.3 ###########################
gaussian_kernel(b) = x->pdf(Normal(0,b), x)

function kernel_density_estimate(ϕ, O)
    return x -> sum([ϕ(x - o) for o in O])/length(O)
end
## END ALGORITHM 4.3 #############################

## BEGIN ALGORITHM 5.1 ###########################
function bayesian_score_component(M, α)
    p = sum(loggamma.(α + M))
    p -= sum(loggamma.(α))
    p += sum(loggamma.(sum(α,dims=2)))
    p -= sum(loggamma.(sum(α,dims=2) + sum(M,dims=2)))
    return p
end

function bayesian_score(vars, G, D)
    n = length(vars)
    M = statistics(vars, G, D)
    α = prior(vars, G)
    return sum(bayesian_score_component(M[i], α[i]) for i in 1:n)
end
## END ALGORITHM 5.1 #############################

## BEGIN ALGORITHM 5.2 ###########################
struct K2Search
    ordering::Vector{Int} # variable ordering
end

function fit(method::K2Search, vars, D)
    G = SimpleDiGraph(length(vars))
    for (k,i) in enumerate(method.ordering[2:end])
        y = bayesian_score(vars, G, D)
        while true
            y_best, j_best = -Inf, 0
            for j in method.ordering[1:k]
                if !has_edge(G, j, i)
                    add_edge!(G, j, i)
                    y′ = bayesian_score(vars, G, D)
                    if y′ > y_best
                        y_best, j_best = y′, j
                    end
                    rem_edge!(G, j, i)
                end
            end
            if y_best > y
                y = y_best
                add_edge!(G, j_best, i)
            else
                break
            end
        end
    end
    return G
end
## END ALGORITHM 5.2 #############################

## BEGIN ALGORITHM 5.3 ###########################
struct LocalDirectedGraphSearch
    G # initial graph
    k_max # number of iterations
end

function rand_graph_neighbor(G)
    n = nv(G)
    i = rand(1:n)
    j = mod1(i + rand(2:n)-1, n)
    G′ = copy(G)
    has_edge(G, i, j) ? rem_edge!(G′, i, j) : add_edge!(G′, i, j)
    return G′
end

function fit(method::LocalDirectedGraphSearch, vars, D)
    G = method.G
    y = bayesian_score(vars, G, D)
    for k in 1:method.k_max
        G′ = rand_graph_neighbor(G)
        y′ = is_cyclic(G′) ? -Inf : bayesian_score(vars, G′, D)
        if y′ > y
            y, G = y′, G′
        end
    end
    return G
end
## END ALGORITHM 5.3 #############################

## BEGIN ALGORITHM 5.4 ###########################
function are_markov_equivalent(G, H)
    if nv(G) != nv(H) || ne(G) != ne(H) || !all(has_edge(H, e) || has_edge(H, reverse(e)) for e in edges(G))
        return false
    end
    for c in 1:nv(G)
        parents = inneighbors(G, c)
        for (a, b) in subsets(parents, 2)
            if !has_edge(G, a, b) && !has_edge(G, b, a) &&
                !(has_edge(H, a, c) && has_edge(H, b, c))
                return false
            end
        end
    end
    return true
end
## END ALGORITHM 5.4 #############################

## BEGIN ALGORITHM 6.1 ###########################
struct SimpleProblem
    bn::BayesianNetwork
    chance_vars::Vector{Variable}
    decision_vars::Vector{Variable}
    utility_vars::Vector{Variable}
    utilities::Dict{Symbol, Vector{Float64}}
end

function solve(P::SimpleProblem, evidence, M)
    query = [var.name for var in P.utility_vars]
    U(a) = sum(P.utilities[uname][a[uname]] for uname in query)
    best = (a=nothing, u=-Inf)
    for assignment in assignments(P.decision_vars)
        evidence = merge(evidence, assignment)
        ϕ = infer(M, P.bn, query, evidence)
        u = sum(p*U(a) for (a, p) in ϕ.table)
        if u > best.u
            best = (a=assignment, u=u)
        end
    end
    return best
end
## END ALGORITHM 6.1 #############################

## BEGIN ALGORITHM 6.2 ###########################
function value_of_information(P, query, evidence, M)
    ϕ = infer(M, P.bn, query, evidence)
    voi = -solve(P, evidence, M).u
    query_vars = filter(v->v.name ∈ query, P.chance_vars)
    for o′ in assignments(query_vars)
        oo′ = merge(evidence, o′)
        p = ϕ.table[o′]
        voi += p*solve(P, oo′, M).u
    end
    return voi
end
## END ALGORITHM 6.2 #############################

## BEGIN ALGORITHM 7.1 ###########################
struct MDP
    γ # discount factor
    S # state space
    A # action space
    T # transition function
    R # reward function
    TR # sample transition and reward
end
## END ALGORITHM 7.1 #############################

## BEGIN ALGORITHM 7.2 ###########################
function lookahead(P::MDP, U, s, a)
    S, T, R, γ = P.S, P.T, P.R, P.γ
    return R(s,a) + γ*sum(T(s,a,s′)*U(s′) for s′ in S)
end

function lookahead(P::MDP, U::Vector, s, a)
    S, T, R, γ = P.S, P.T, P.R, P.γ
    return R(s,a) + γ*sum(T(s,a,s′)*U[i] for (i,s′) in enumerate(S))
end
## END ALGORITHM 7.2 #############################

## BEGIN ALGORITHM 7.3 ###########################
function iterative_policy_evaluation(P::MDP, π, k_max)
    S, T, R, γ = P.S, P.T, P.R, P.γ
    U = [0.0 for s in S]
    for k in 1:k_max
        U = [lookahead(P, U, s, π(s)) for s in S]
    end
    return U
end
## END ALGORITHM 7.3 #############################

## BEGIN ALGORITHM 7.4 ###########################
function policy_evaluation(P::MDP, π)
    S, T, R, γ = P.S, P.T, P.R, P.γ
    R′ = [R(s, π(s)) for s in S]
    T′ = [T(s, π(s), s′) for s in S, s′ in S]
    return (I - γ*T′)\R′
end
## END ALGORITHM 7.4 #############################

## BEGIN ALGORITHM 7.5 ###########################
struct ValueFunctionPolicy
    P # problem
    U # utility function
end

function greedy(P::MDP, U, s)
    u, a = findmax(a->lookahead(P, U, s, a), P.A)
    return (a=a, u=u)
end

(π::ValueFunctionPolicy)(s) = greedy(π.P, π.U, s).a
## END ALGORITHM 7.5 #############################

## BEGIN ALGORITHM 7.6 ###########################
struct PolicyIteration
    π # initial policy
    k_max # maximum number of iterations
end

function solve(M::PolicyIteration, P::MDP)
    π, S = M.π, P.S
    for k = 1:M.k_max
        U = policy_evaluation(P, π)
        π′ = ValueFunctionPolicy(P, U)
        if all(π(s) == π′(s) for s in S)
            break
        end
        π = π′
    end
    return π
end
## END ALGORITHM 7.6 #############################

## BEGIN ALGORITHM 7.7 ###########################
function backup(P::MDP, U, s)
    return maximum(lookahead(P, U, s, a) for a in P.A)
end
## END ALGORITHM 7.7 #############################

## BEGIN ALGORITHM 7.8 ###########################
struct ValueIteration
    k_max # maximum number of iterations
end

function solve(M::ValueIteration, P::MDP)
    U = [0.0 for s in P.S]
    for k = 1:M.k_max
        U = [backup(P, U, s) for s in P.S]
    end
    return ValueFunctionPolicy(P, U)
end
## END ALGORITHM 7.8 #############################

## BEGIN ALGORITHM 7.9 ###########################
struct GaussSeidelValueIteration
    k_max # maximum number of iterations
end

function solve(M::GaussSeidelValueIteration, P::MDP)
    U = [0.0 for s in S]
    for k = 1:M.k_max
        for (s, i) in enumerate(P.S)
            U[i] = backup(P, U, s)
        end
    end
    return ValueFunctionPolicy(P, U)
end
## END ALGORITHM 7.9 #############################

## BEGIN ALGORITHM 7.10 ###########################
struct LinearProgramFormulation end

function tensorform(P::MDP)
    S, A, R, T = P.S, P.A, P.R, P.T
    S_prime = eachindex(S)
    A_prime = eachindex(A)
    R_prime = [R(s,a) for s in S, a in A]
    T_prime = [T(s,a,s_prime) for s in S, a in A, s_prime in S]
    return S_prime, A_prime, R_prime, T_prime
end

solve(P::MDP) = solve(LinearProgramFormulation(), P)

function solve(M::LinearProgramFormulation, P::MDP)
    S, A, R, T = tensorform(P)
    model = Model(GLPK.Optimizer)
    @variable(model, U[S])
    @objective(model, Min, sum(U))
    @constraint(model, [s=S, a=A], U[s] ≥ R[s,a] + P.γ*T[s,a,:]⋅U)
    optimize!(model)
    return ValueFunctionPolicy(P, value.(U))
end
## END ALGORITHM 7.10 #############################

## BEGIN ALGORITHM 7.11 ###########################
struct LinearQuadraticProblem
    Ts # transition matrix with respect to state
    Ta # transition matrix with respect to action
    Rs # reward matrix with respect to state (negative semidefinite)
    Ra # reward matrix with respect to action (negative definite)
    h_max # horizon
end

function solve(P::LinearQuadraticProblem)
    Ts, Ta, Rs, Ra, h_max = P.Ts, P.Ta, P.Rs, P.Ra, P.h_max
    V = zeros(size(Rs))
    πs = Any[s -> zeros(size(Ta, 2))]
    for h in 2:h_max
        V = Ts'*(V - V*Ta*((Ta'*V*Ta + Ra) \ Ta'*V))*Ts + Rs
        L = -(Ta'*V*Ta + Ra) \ Ta' * V * Ts
        push!(πs, s -> L*s)
    end
    return πs
end
## END ALGORITHM 7.11 #############################

## BEGIN ALGORITHM 8.1 ###########################
struct ApproximateValueIteration
    Uθ # initial parameterized value function that supports fit!
    S # set of discrete states for performing backups
    k_max # maximum number of iterations
end

function solve(M::ApproximateValueIteration, P::MDP)
    Uθ, S, k_max = M.Uθ, M.S, M.k_max
    for k in 1:k_max
        U = [backup(P, Uθ, s) for s in S]
        fit!(Uθ, S, U)
    end
    return ValueFunctionPolicy(P, Uθ)
end
## END ALGORITHM 8.1 #############################

## BEGIN ALGORITHM 8.2 ###########################
mutable struct NearestNeighborValueFunction
    k # number of neighbors
    d # distance function d(s, s′)
    S # set of discrete states
    θ # vector of values at states in S
end

function (Uθ::NearestNeighborValueFunction)(s)
    dists = [Uθ.d(s,s′) for s′ in Uθ.S]
    ind = sortperm(dists)[1:Uθ.k]
    return mean(Uθ.θ[i] for i in ind)
end

function fit!(Uθ::NearestNeighborValueFunction, S, U)
    Uθ.θ = U
    return Uθ
end
## END ALGORITHM 8.2 #############################

## BEGIN ALGORITHM 8.3 ###########################
mutable struct LocallyWeightedValueFunction
    k # kernel function k(s, s′)
    S # set of discrete states
    θ # vector of values at states in S
end

function (Uθ::LocallyWeightedValueFunction)(s)
    w = normalize([Uθ.k(s,s′) for s′ in Uθ.S], 1)
    return Uθ.θ ⋅ w
end

function fit!(Uθ::LocallyWeightedValueFunction, S, U)
    Uθ.θ = U
    return Uθ
end
## END ALGORITHM 8.3 #############################

## BEGIN ALGORITHM 8.4 ###########################
mutable struct MultilinearValueFunction
    o # position of lower-left corner
    δ # vector of widths
    θ # vector of values at states in S
end

function (Uθ::MultilinearValueFunction)(s)
    o, δ, θ = Uθ.o, Uθ.δ, Uθ.θ
    Δ = (s - o)./δ
    # Multidimensional index of lower-left cell
    i = min.(floor.(Int, Δ) .+ 1, size(θ) .- 1)
    vertex_index = similar(i)
    d = length(s)
    u = 0.0
    for vertex in 0:2^d-1
        weight = 1.0
        for j in 1:d
            # Check whether jth bit is set
            if vertex & (1 << (j-1)) > 0
                vertex_index[j] = i[j] + 1
                weight *= Δ[j] - i[j] + 1
            else
                vertex_index[j] = i[j]
                weight *= i[j] - Δ[j]
            end
        end
        u += θ[vertex_index...]*weight
    end
    return u
end

function fit!(Uθ::MultilinearValueFunction, S, U)
    Uθ.θ = U
    return Uθ
end
## END ALGORITHM 8.4 #############################

## BEGIN ALGORITHM 8.5 ###########################
mutable struct SimplexValueFunction
    o # position of lower-left corner
    δ # vector of widths
    θ # vector of values at states in S
end

function (Uθ::SimplexValueFunction)(s)
    Δ = (s - Uθ.o)./Uθ.δ
    # Multidimensional index of upper-right cell
    i = min.(floor.(Int, Δ) .+ 1, size(Uθ.θ) .- 1) .+ 1
    u = 0.0
    s′ = (s - (Uθ.o + Uθ.δ.*(i.-2))) ./ Uθ.δ
    p = sortperm(s′) # increasing order
    w_tot = 0.0
    for j in p
        w = s′[j] - w_tot
        u += w*Uθ.θ[i...]
        i[j] -= 1
        w_tot += w
    end
    u += (1 - w_tot)*Uθ.θ[i...]
    return u
end

function fit!(Uθ::SimplexValueFunction, S, U)
    Uθ.θ = U
    return Uθ
end
## END ALGORITHM 8.5 #############################

## BEGIN ALGORITHM 8.6 ###########################
mutable struct LinearRegressionValueFunction
    β # basis vector function
    θ # vector of parameters
end

function (Uθ::LinearRegressionValueFunction)(s)
    return Uθ.β(s) ⋅ Uθ.θ
end

function fit!(Uθ::LinearRegressionValueFunction, S, U)
    X = hcat([Uθ.β(s) for s in S]...)'
    Uθ.θ = pinv(X)*U
    return Uθ
end
## END ALGORITHM 8.6 #############################

## BEGIN ALGORITHM 9.1 ###########################
struct RolloutLookahead
    P # problem
    π # rollout policy
    d # depth
end

randstep(P::MDP, s, a) = P.TR(s, a)

function rollout(P, s, π, d)
    ret = 0.0
    for t in 1:d
        a = π(s)
        s, r = randstep(P, s, a)
        ret += P.γ^(t-1) * r
    end
    return ret
end

function (π::RolloutLookahead)(s)
    U(s) = rollout(π.P, s, π.π, π.d)
    return greedy(π.P, U, s).a
end
## END ALGORITHM 9.1 #############################

## BEGIN ALGORITHM 9.2 ###########################
struct ForwardSearch
    P # problem
    d # depth
    U # value function at depth d
end

function forward_search(P, s, d, U)
    if d ≤ 0
        return (a=nothing, u=U(s))
    end
    best = (a=nothing, u=-Inf)
    U′(s) = forward_search(P, s, d-1, U).u
    for a in P.A
        u = lookahead(P, U′, s, a)
        if u > best.u
            best = (a=a, u=u)
        end
    end
    return best
end

(π::ForwardSearch)(s) = forward_search(π.P, s, π.d, π.U).a
## END ALGORITHM 9.2 #############################

## BEGIN ALGORITHM 9.3 ###########################
struct BranchAndBound
    P # problem
    d # depth
    Ulo # lower bound on value function at depth d
    Qhi # upper bound on action value function
end

function branch_and_bound(P, s, d, Ulo, Qhi)
    if d ≤ 0
        return (a=nothing, u=Ulo(s))
    end
    U′(s) = branch_and_bound(P, s, d-1, Ulo, Qhi).u
    best = (a=nothing, u=-Inf)
    for a in sort(P.A, by=a->Qhi(s,a), rev=true)
        if Qhi(s, a) < best.u
            return best # safe to prune
        end
        u = lookahead(P, U′, s, a)
        if u > best.u
            best = (a=a, u=u)
        end
    end
    return best
end

(π::BranchAndBound)(s) = branch_and_bound(π.P, s, π.d, π.Ulo, π.Qhi).a
## END ALGORITHM 9.3 #############################

## BEGIN ALGORITHM 9.4 ###########################
struct SparseSampling
    P # problem
    d # depth
    m # number of samples
    U # value function at depth d
end

function sparse_sampling(P, s, d, m, U)
    if d ≤ 0
        return (a=nothing, u=U(s))
    end
    best = (a=nothing, u=-Inf)
    for a in P.A
        u = 0.0
        for i in 1:m
            s′, r = randstep(P, s, a)
            a′, u′ = sparse_sampling(P, s′, d-1, m, U)
            u += (r + P.γ*u′) / m
        end
        if u > best.u
            best = (a=a, u=u)
        end
    end
    return best
end

(π::SparseSampling)(s) = sparse_sampling(π.P, s, π.d, π.m, π.U).a
## END ALGORITHM 9.4 #############################

## BEGIN ALGORITHM 9.5 ###########################
struct MonteCarloTreeSearch
    P # problem
    N # visit counts
    Q # action value estimates
    d # depth
    m # number of simulations
    c # exploration constant
    U # value function estimate
end

function (π::MonteCarloTreeSearch)(s)
    for k in 1:π.m
        simulate!(π, s)
    end
    return argmax(a->π.Q[(s,a)], π.P.A)
end
## END ALGORITHM 9.5 #############################

## BEGIN ALGORITHM 9.6 ###########################
function simulate!(π::MonteCarloTreeSearch, s, d=π.d)
    if d ≤ 0
        return π.U(s)
    end
    P, N, Q, c = π.P, π.N, π.Q, π.c
    A, TR, γ = P.A, P.TR, P.γ
    if !haskey(N, (s, first(A)))
        for a in A
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
        end
        return π.U(s)
    end
    a = explore(π, s)
    s′, r = TR(s,a)
    q = r + γ*simulate!(π, s′, d-1)
    N[(s,a)] += 1
    Q[(s,a)] += (q-Q[(s,a)])/N[(s,a)]
    return q
end
## END ALGORITHM 9.6 #############################

## BEGIN ALGORITHM 9.7 ###########################
bonus(Nsa, Ns) = Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa)

function explore(π::MonteCarloTreeSearch, s)
    A, N, Q, c = π.P.A, π.N, π.Q, π.c
    Ns = sum(N[(s,a)] for a in A)
    return argmax(a->Q[(s,a)] + c*bonus(N[(s,a)], Ns), A)
end
## END ALGORITHM 9.7 #############################

## BEGIN ALGORITHM 9.8 ###########################
struct HeuristicSearch
    P # problem
    Uhi # upper bound on value function
    d # depth
    m # number of simulations
end

function simulate!(π::HeuristicSearch, U, s)
    P = π.P
    for d in 1:π.d
        a, u = greedy(P, U, s)
        U[s] = u
        s = rand(P.T(s, a))
    end
end

function (π::HeuristicSearch)(s)
    U = [π.Uhi(s) for s in π.P.S]
    for i in 1:m
        simulate!(π, U, s)
    end
    return greedy(π.P, U, s).a
end
## END ALGORITHM 9.8 #############################

## BEGIN ALGORITHM 9.9 ###########################
struct LabeledHeuristicSearch
    P # problem
    Uhi # upper bound on value function
    d # depth
    δ # gap threshold
end

function (π::LabeledHeuristicSearch)(s)
    U, solved = [π.Uhi(s) for s in P.S], Set()
    while s ∉ solved
        simulate!(π, U, solved, s)
    end
    return greedy(π.P, U, s).a
end
## END ALGORITHM 9.9 #############################

## BEGIN ALGORITHM 9.10 ###########################
function simulate!(π::LabeledHeuristicSearch, U, solved, s)
    visited = []
    for d in 1:π.d
        if s ∈ solved
            break
        end
        push!(visited, s)
        a, u = greedy(π.P, U, s)
        U[s] = u
        s = rand(π.P.T(s, a))
    end
    while !isempty(visited)
        if label!(π, U, solved, pop!(visited))
            break
        end
    end
end
## END ALGORITHM 9.10 #############################

## BEGIN ALGORITHM 9.11 ###########################
function expand(π::LabeledHeuristicSearch, U, solved, s)
    P, δ = π.P, π.δ
    S, A, T = P.S, P.A, P.T
    found, toexpand, envelope = false, Set(s), []
    while !isempty(toexpand)
        s = pop!(toexpand)
        push!(envelope, s)
        a, u = greedy(P, U, s)
        if abs(U[s] - u) > δ
            found = true
        else
            for s′ in S
                if T(s,a,s′) > 0 && s′ ∉ (solved ∪ envelope)
                    push!(toexpand, s′)
                end
            end
        end
    end
    return (found, envelope)
end

function label!(π::LabeledHeuristicSearch, U, solved, s)
    if s ∈ solved
        return false
    end
    found, envelope = expand(π, U, solved, s)
    if found
        for s ∈ reverse(envelope)
            U[s] = greedy(π.P, U, s).u
        end
    else
        union!(solved, envelope)
    end
    return found
end
## END ALGORITHM 9.11 #############################

## BEGIN ALGORITHM 10.1 ###########################
struct MonteCarloPolicyEvaluation
    P # problem
    b # initial state distribution
    d # depth
    m # number of samples
end

function (U::MonteCarloPolicyEvaluation)(π)
    R(π) = rollout(U.P, rand(U.b), π, U.d)
    return mean(R(π) for i = 1:U.m)
end

(U::MonteCarloPolicyEvaluation)(π, θ) = U(s->π(θ, s))
## END ALGORITHM 10.1 #############################

## BEGIN ALGORITHM 10.2 ###########################
struct HookeJeevesPolicySearch
    θ # initial parameterization
    α # step size
    c # step size reduction factor
    ϵ # termination step size
end

function optimize(M::HookeJeevesPolicySearch, π, U)
    θ, θ′, α, c, ϵ = copy(M.θ), similar(M.θ), M.α, M.c, M.ϵ
    u, n = U(π, θ), length(θ)
    while α > ϵ
        copyto!(θ′, θ)
        best = (i=0, sgn=0, u=u)
        for i in 1:n
            for sgn in (-1,1)
                θ′[i] = θ[i] + sgn*α
                u′ = U(π, θ′)
                if u′ > best.u
                    best = (i=i, sgn=sgn, u=u′)
                end
            end
            θ′[i] = θ[i]
        end
        if best.i != 0
            θ[best.i] += best.sgn*α
            u = best.u
        else
            α *= c
        end
    end
    return θ
end
## END ALGORITHM 10.2 #############################

## BEGIN ALGORITHM 10.3 ###########################
struct GeneticPolicySearch
    θs # initial population
    σ # initial standard devidation
    m_elite # number of elite samples
    k_max # number of iterations
end

function optimize(M::GeneticPolicySearch, π, U)
    θs, σ = M.θs, M.σ
    n, m = length(first(θs)), length(θs)
    for k in 1:M.k_max
        us = [U(π, θ) for θ in θs]
        sp = sortperm(us, rev=true)
        θ_best = θs[sp[1]]
        rand_elite() = θs[sp[rand(1:M.m_elite)]]
        θs = [rand_elite() + σ.*randn(n) for i in 1:(m-1)]
        push!(θs, θ_best)
    end
    return last(θs)
end
## END ALGORITHM 10.3 #############################

## BEGIN ALGORITHM 10.4 ###########################
struct CrossEntropyPolicySearch
    p # initial distribution
    m # number of samples
    m_elite # number of elite samples
    k_max # number of iterations
end

function optimize_dist(M::CrossEntropyPolicySearch, π, U)
    p, m, m_elite, k_max = M.p, M.m, M.m_elite, M.k_max
    for k in 1:k_max
        θs = rand(p, m)
        us = [U(π, θs[:,i]) for i in 1:m]
        θ_elite = θs[:,sortperm(us)[(m-m_elite+1):m]]
        p = Distributions.fit(typeof(p), θ_elite)
    end
    return p
end

function optimize(M, π, U)
    return Distributions.mode(optimize_dist(M, π, U))
end
## END ALGORITHM 10.4 #############################

## BEGIN ALGORITHM 10.5 ###########################
struct EvolutionStrategies
    D # distribution constructor
    ψ # initial distribution parameterization
    ∇logp # log search likelihood gradient
    m # number of samples
    α # step factor
    k_max # number of iterations
end

function evolution_strategy_weights(m)
    ws = [max(0, log(m/2+1) - log(i)) for i in 1:m]
    ws ./= sum(ws)
    ws .-= 1/m
    return ws
end

function optimize_dist(M::EvolutionStrategies, π, U)
    D, ψ, m, ∇logp, α = M.D, M.ψ, M.m, M.∇logp, M.α
    ws = evolution_strategy_weights(m)
    for k in 1:M.k_max
        θs = rand(D(ψ), m)
        us = [U(π, θs[:,i]) for i in 1:m]
        sp = sortperm(us, rev=true)
        ∇ = sum(w.*∇logp(ψ, θs[:,i]) for (w,i) in zip(ws,sp))
        ψ += α.*∇
    end
    return D(ψ)
end
## END ALGORITHM 10.5 #############################

## BEGIN ALGORITHM 10.6 ###########################
struct IsotropicEvolutionStrategies
    ψ # initial mean
    σ # initial standard devidation
    m # number of samples
    α # step factor
    k_max # number of iterations
end

function optimize_dist(M::IsotropicEvolutionStrategies, π, U)
    ψ, σ, m, α, k_max = M.ψ, M.σ, M.m, M.α, M.k_max
    n = length(ψ)
    ws = evolution_strategy_weights(2*div(m,2))
    for k in 1:k_max
        ϵs = [randn(n) for i in 1:div(m,2)]
        append!(ϵs, -ϵs) # weight mirroring
        us = [U(π, ψ + σ.*ϵ) for ϵ in ϵs]
        sp = sortperm(us, rev=true)
        ∇ = sum(w.*ϵs[i] for (w,i) in zip(ws,sp)) / σ
        ψ += α.*∇
    end
    return MvNormal(ψ, σ)
end
## END ALGORITHM 10.6 #############################

## BEGIN ALGORITHM 11.1 ###########################
function simulate(P::MDP, s, π, d)
    τ = []
    for i = 1:d
        a = π(s)
        s′, r = P.TR(s,a)
        push!(τ, (s,a,r))
        s = s′
    end
    return τ
end
## END ALGORITHM 11.1 #############################

## BEGIN ALGORITHM 11.2 ###########################
struct FiniteDifferenceGradient
    P # problem
    b # initial state distribution
    d # depth
    m # number of samples
    δ # step size
end

function gradient(M::FiniteDifferenceGradient, π, θ)
    P, b, d, m, δ, γ, n = M.P, M.b, M.d, M.m, M.δ, M.P.γ, length(θ)
    Δθ(i) = [i == k ? δ : 0.0 for k in 1:n]
    R(τ) = sum(r*γ^(k-1) for (k, (s,a,r)) in enumerate(τ))
    U(θ) = mean(R(simulate(P, rand(b), s->π(θ, s), d)) for i in 1:m)
    ΔU = [U(θ + Δθ(i)) - U(θ) for i in 1:n]
    return ΔU ./ δ
end
## END ALGORITHM 11.2 #############################

## BEGIN ALGORITHM 11.3 ###########################
struct RegressionGradient
    P # problem
    b # initial state distribution
    d # depth
    m # number of samples
    δ # step size
end

function gradient(M::RegressionGradient, π, θ)
    P, b, d, m, δ, γ = M.P, M.b, M.d, M.m, M.δ, M.P.γ
    ΔΘ = [δ.*normalize(randn(length(θ)), 2) for i = 1:m]
    R(τ) = sum(r*γ^(k-1) for (k, (s,a,r)) in enumerate(τ))
    U(θ) = R(simulate(P, rand(b), s->π(θ,s), d))
    ΔU = [U(θ + Δθ) - U(θ) for Δθ in ΔΘ]
    return pinv(reduce(hcat, ΔΘ)') * ΔU
end
## END ALGORITHM 11.3 #############################

## BEGIN ALGORITHM 11.4 ###########################
struct LikelihoodRatioGradient
    P # problem
    b # initial state distribution
    d # depth
    m # number of samples
    ∇logπ # gradient of log likelihood
end

function gradient(M::LikelihoodRatioGradient, π, θ)
    P, b, d, m, ∇logπ, γ = M.P, M.b, M.d, M.m, M.∇logπ, M.P.γ
    πθ(s) = π(θ, s)
    R(τ) = sum(r*γ^(k-1) for (k, (s,a,r)) in enumerate(τ))
    ∇U(τ) = sum(∇logπ(θ, a, s) for (s,a) in τ)*R(τ)
    return mean(∇U(simulate(P, rand(b), πθ, d)) for i in 1:m)
end
## END ALGORITHM 11.4 #############################

## BEGIN ALGORITHM 11.5 ###########################
struct RewardToGoGradient
    P # problem
    b # initial state distribution
    d # depth
    m # number of samples
    ∇logπ # gradient of log likelihood
end

function gradient(M::RewardToGoGradient, π, θ)
    P, b, d, m, ∇logπ, γ = M.P, M.b, M.d, M.m, M.∇logπ, M.P.γ
    πθ(s) = π(θ, s)
    R(τ, j) = sum(r*γ^(k-1) for (k,(s,a,r)) in zip(j:d, τ[j:end]))
    ∇U(τ) = sum(∇logπ(θ, a, s)*R(τ,j) for (j, (s,a,r)) in enumerate(τ))
    return mean(∇U(simulate(P, rand(b), πθ, d)) for i in 1:m)
end
## END ALGORITHM 11.5 #############################

## BEGIN ALGORITHM 11.6 ###########################
struct BaselineSubtractionGradient
    P # problem
    b # initial state distribution
    d # depth
    m # number of samples
    ∇logπ # gradient of log likelihood
end

function gradient(M::BaselineSubtractionGradient, π, θ)
    P, b, d, m, ∇logπ, γ = M.P, M.b, M.d, M.m, M.∇logπ, M.P.γ
    πθ(s) = π(θ, s)
    ℓ(a, s, k) = ∇logπ(θ, a, s)*γ^(k-1)
    R(τ, k) = sum(r*γ^(j-1) for (j,(s,a,r)) in enumerate(τ[k:end]))
    numer(τ) = sum(ℓ(a,s,k).^2*R(τ,k) for (k,(s,a,r)) in enumerate(τ))
    denom(τ) = sum(ℓ(a,s,k).^2 for (k,(s,a)) in enumerate(τ))
    base(τ) = numer(τ) ./ denom(τ)
    trajs = [simulate(P, rand(b), πθ, d) for i in 1:m]
    rbase = mean(base(τ) for τ in trajs)
    ∇U(τ) = sum(ℓ(a,s,k).*(R(τ,k).-rbase) for (k,(s,a,r)) in enumerate(τ))
    return mean(∇U(τ) for τ in trajs)
end
## END ALGORITHM 11.6 #############################

## BEGIN ALGORITHM 12.1 ###########################
struct PolicyGradientUpdate
    ∇U # policy gradient estimate
    α # step factor
end

function update(M::PolicyGradientUpdate, θ)
    return θ + M.α * M.∇U(θ)
end
## END ALGORITHM 12.1 #############################

## BEGIN ALGORITHM 12.2 ###########################
scale_gradient(∇, L2_max) = min(L2_max/norm(∇), 1)*∇
clip_gradient(∇, a, b) = clamp.(∇, a, b)
## END ALGORITHM 12.2 #############################

## BEGIN ALGORITHM 12.3 ###########################
struct RestrictedPolicyUpdate
    P # problem
    b # initial state distribution
    d # depth
    m # number of samples
    ∇logπ # gradient of log likelihood
    π # policy
    ϵ # divergence bound
end

function update(M::RestrictedPolicyUpdate, θ)
    P, b, d, m, ∇logπ, π, γ = M.P, M.b, M.d, M.m, M.∇logπ, M.π, M.P.γ
    πθ(s) = π(θ, s)
    R(τ) = sum(r*γ^(k-1) for (k, (s,a,r)) in enumerate(τ))
    τs = [simulate(P, rand(b), πθ, d) for i in 1:m]
    ∇log(τ) = sum(∇logπ(θ, a, s) for (s,a) in τ)
    ∇U(τ) = ∇log(τ)*R(τ)
    u = mean(∇U(τ) for τ in τs)
    return θ + u*sqrt(2*M.ϵ/dot(u,u))
end
## END ALGORITHM 12.3 #############################

## BEGIN ALGORITHM 12.4 ###########################
struct NaturalPolicyUpdate
    P # problem
    b # initial state distribution
    d # depth
    m # number of samples
    ∇logπ # gradient of log likelihood
    π # policy
    ϵ # divergence bound
end

function natural_update(θ, ∇f, F, ϵ, τs)
    ∇fθ = mean(∇f(τ) for τ in τs)
    u = mean(F(τ) for τ in τs) \ ∇fθ
    return θ + u*sqrt(2ϵ/dot(∇fθ,u))
end

function update(M::NaturalPolicyUpdate, θ)
    P, b, d, m, ∇logπ, π, γ = M.P, M.b, M.d, M.m, M.∇logπ, M.π, M.P.γ
    πθ(s) = π(θ, s)
    R(τ) = sum(r*γ^(k-1) for (k, (s,a,r)) in enumerate(τ))
    ∇log(τ) = sum(∇logπ(θ, a, s) for (s,a) in τ)
    ∇U(τ) = ∇log(τ)*R(τ)
    F(τ) = ∇log(τ)*∇log(τ)'
    τs = [simulate(P, rand(b), πθ, d) for i in 1:m]
    return natural_update(θ, ∇U, F, M.ϵ, τs)
end
## END ALGORITHM 12.4 #############################

## BEGIN ALGORITHM 12.5 ###########################
struct TrustRegionUpdate
    P # problem
    b # initial state distribution
    d # depth
    m # number of samples
    π # policy π(s)
    p # policy likelihood p(θ, a, s)
    ∇logπ # log likelihood gradient
    KL # KL divergence KL(θ, θ′, s)
    ϵ # divergence bound
    α # line search reduction factor (e.g. 0.5)
end

function surrogate_objective(M::TrustRegionUpdate, θ, θ′, τs)
    d, p, γ = M.d, M.p, M.P.γ
    R(τ, j) = sum(r*γ^(k-1) for (k,(s,a,r)) in zip(j:d, τ[j:end]))
    w(a,s) = p(θ′,a,s) / p(θ,a,s)
    f(τ) = mean(w(a,s)*R(τ,k) for (k,(s,a,r)) in enumerate(τ))
    return mean(f(τ) for τ in τs)
end

function surrogate_constraint(M::TrustRegionUpdate, θ, θ′, τs)
    γ = M.P.γ
    KL(τ) = mean(M.KL(θ, θ′, s)*γ^(k-1) for (k,(s,a,r)) in enumerate(τ))
    return mean(KL(τ) for τ in τs)
end

function linesearch(M::TrustRegionUpdate, f, g, θ, θ′)
    fθ = f(θ)
    while g(θ′) > M.ϵ || f(θ′) ≤ fθ
        θ′ = θ + M.α*(θ′ - θ)
    end
    return θ′
end

function update(M::TrustRegionUpdate, θ)
    P, b, d, m, ∇logπ, π, γ = M.P, M.b, M.d, M.m, M.∇logπ, M.π, M.P.γ
    πθ(s) = π(θ, s)
    R(τ) = sum(r*γ^(k-1) for (k, (s,a,r)) in enumerate(τ))
    ∇log(τ) = sum(∇logπ(θ, a, s) for (s,a) in τ)
    ∇U(τ) = ∇log(τ)*R(τ)
    F(τ) = ∇log(τ)*∇log(τ)'
    τs = [simulate(P, rand(b), πθ, d) for i in 1:m]
    θ′ = natural_update(θ, ∇U, F, M.ϵ, τs)
    f(θ′) = surrogate_objective(M, θ, θ′, τs)
    g(θ′) = surrogate_constraint(M, θ, θ′, τs)
    return linesearch(M, f, g, θ, θ′)
end
## END ALGORITHM 12.5 #############################

## BEGIN ALGORITHM 12.6 ###########################
struct ClampedSurrogateUpdate
    P # problem
    b # initial state distribution
    d # depth
    m # number of trajectories
    π # policy
    p # policy likelihood
    ∇π # policy likelihood gradient
    ϵ # divergence bound
    α # step size
    k_max # number of iterations per update
end

function clamped_gradient(M::ClampedSurrogateUpdate, θ, θ′, τs)
    d, p, ∇π, ϵ, γ = M.d, M.p, M.∇π, M.ϵ, M.P.γ
    R(τ, j) = sum(r*γ^(k-1) for (k,(s,a,r)) in zip(j:d, τ[j:end]))
    ∇f(a,s,r_togo) = begin
        P = p(θ, a,s)
        w = p(θ′,a,s) / P
        if (r_togo > 0 && w > 1+ϵ) || (r_togo < 0 && w < 1-ϵ)
            return zeros(length(θ))
        end
        return ∇π(θ′, a, s) * r_togo / P
    end
    ∇f(τ) = mean(∇f(a,s,R(τ,k)) for (k,(s,a,r)) in enumerate(τ))
    return mean(∇f(τ) for τ in τs)
end

function update(M::ClampedSurrogateUpdate, θ)
    P, b, d, m, π, α, k_max= M.P, M.b, M.d, M.m, M.π, M.α, M.k_max
    πθ(s) = π(θ, s)
    τs = [simulate(P, rand(b), πθ, d) for i in 1:m]
    θ′ = copy(θ)
    for k in 1:k_max
        θ′ += α*clamped_gradient(M, θ, θ′, τs)
    end
    return θ′
end
## END ALGORITHM 12.6 #############################

## BEGIN ALGORITHM 13.1 ###########################
struct ActorCritic
    P # problem
    b # initial state distribution
    d # depth
    m # number of samples
    ∇logπ # gradient of log likelihood ∇logπ(θ,a,s)
    U # parameterized value function U(ϕ, s)
    ∇U # gradient of value function ∇U(ϕ,s)
end
function gradient(M::ActorCritic, π, θ, ϕ)
    P, b, d, m, ∇logπ = M.P, M.b, M.d, M.m, M.∇logπ
    U, ∇U, γ = M.U, M.∇U, M.P.γ
    πθ(s) = π(θ, s)
    R(τ,j) = sum(r*γ^(k-1) for (k,(s,a,r)) in enumerate(τ[j:end]))
    A(τ,j) = τ[j][3] + γ*U(ϕ,τ[j+1][1]) - U(ϕ,τ[j][1])
    ∇Uθ(τ) = sum(∇logπ(θ,a,s)*A(τ,j)*γ^(j-1) for (j, (s,a,r)) in enumerate(τ[1:end-1]))
    ∇ℓϕ(τ) = sum((U(ϕ,s) - R(τ,j))*∇U(ϕ,s) for (j, (s,a,r)) in enumerate(τ))
    trajs = [simulate(P, rand(b), πθ, d) for i in 1:m]
    return mean(∇Uθ(τ) for τ in trajs), mean(∇ℓϕ(τ) for τ in trajs)
end
## END ALGORITHM 13.1 #############################

## BEGIN ALGORITHM 13.2 ###########################
struct GeneralizedAdvantageEstimation
    P # problem
    b # initial state distribution
    d # depth
    m # number of samples
    ∇logπ # gradient of log likelihood ∇logπ(θ,a,s)
    U # parameterized value function U(ϕ, s)
    ∇U # gradient of value function ∇U(ϕ,s)
    λ # weight ∈ [0,1]
end
function gradient(M::GeneralizedAdvantageEstimation, π, θ, ϕ)
    P, b, d, m, ∇logπ = M.P, M.b, M.d, M.m, M.∇logπ
    U, ∇U, γ, λ = M.U, M.∇U, M.P.γ, M.λ
    πθ(s) = π(θ, s)
    R(τ,j) = sum(r*γ^(k-1) for (k,(s,a,r)) in enumerate(τ[j:end]))
    δ(τ,j) = τ[j][3] + γ*U(ϕ,τ[j+1][1]) - U(ϕ,τ[j][1])
    A(τ,j) = sum((γ*λ)^(ℓ-1)*δ(τ, j+ℓ-1) for ℓ in 1:d-j)
    ∇Uθ(τ) = sum(∇logπ(θ,a,s)*A(τ,j)*γ^(j-1) for (j, (s,a,r)) in enumerate(τ[1:end-1]))
    ∇ℓϕ(τ) = sum((U(ϕ,s) - R(τ,j))*∇U(ϕ,s) for (j, (s,a,r)) in enumerate(τ))
    trajs = [simulate(P, rand(b), πθ, d) for i in 1:m]
    return mean(∇Uθ(τ) for τ in trajs), mean(∇ℓϕ(τ) for τ in trajs)
end
## END ALGORITHM 13.2 #############################

## BEGIN ALGORITHM 13.3 ###########################
struct DeterministicPolicyGradient
    P # problem
    b # initial state distribution
    d # depth
    m # number of samples
    ∇π # gradient of deterministic policy π(θ, s)
    Q # parameterized value function Q(ϕ,s,a)
    ∇Qϕ # gradient of value function with respect to ϕ
    ∇Qa # gradient of value function with respect to a
    σ # policy noise
end
function gradient(M::DeterministicPolicyGradient, π, θ, ϕ)
    P, b, d, m, ∇π = M.P, M.b, M.d, M.m, M.∇π
    Q, ∇Qϕ, ∇Qa, σ, γ = M.Q, M.∇Qϕ, M.∇Qa, M.σ, M.P.γ
    π_rand(s) = π(θ, s) + σ*randn()*I
    ∇Uθ(τ) = sum(∇π(θ,s)*∇Qa(ϕ,s,π(θ,s))*γ^(j-1) for (j,(s,a,r)) in enumerate(τ))
    ∇ℓϕ(τ,j) = begin
        s, a, r = τ[j]
        s′ = τ[j+1][1]
        a′ = π(θ,s′)
        δ = r + γ*Q(ϕ,s′,a′) - Q(ϕ,s,a)
        return δ*(γ*∇Qϕ(ϕ,s′,a′) - ∇Qϕ(ϕ,s,a))
    end
    ∇ℓϕ(τ) = sum(∇ℓϕ(τ,j) for j in 1:length(τ)-1)
    trajs = [simulate(P, rand(b), π_rand, d) for i in 1:m]
    return mean(∇Uθ(τ) for τ in trajs), mean(∇ℓϕ(τ) for τ in trajs)
end
## END ALGORITHM 13.3 #############################

## BEGIN ALGORITHM 14.1 ###########################
function adversarial(P::MDP, π, λ)
    S, A, T, R, γ = P.S, P.A, P.T, P.R, P.γ
    S′ = A′ = S
    R′ = zeros(length(S′), length(A′))
    T′ = zeros(length(S′), length(A′), length(S′))
    for s in S′
        for a in A′
            R′[s,a] = -R(s, π(s)) + λ*log(T(s, π(s), a))
            T′[s,a,a] = 1
        end
    end
    return MDP(T′, R′, γ)
end
## END ALGORITHM 14.1 #############################

## BEGIN ALGORITHM 15.1 ###########################
struct BanditProblem
    θ # vector of payoff probabilities
    R # reward sampler
end
function BanditProblem(θ)
    R(a) = rand() < θ[a] ? 1 : 0
    return BanditProblem(θ, R)
end
function simulate(P::BanditProblem, model, π, h)
    for i in 1:h
        a = π(model)
        r = P.R(a)
        update!(model, a, r)
    end
end
## END ALGORITHM 15.1 #############################

## BEGIN ALGORITHM 15.2 ###########################
struct BanditModel
    B # vector of beta distributions
end
function update!(model::BanditModel, a, r)
    α, β = StatsBase.params(model.B[a])
    model.B[a] = Beta(α + r, β + (1-r))
    return model
end
## END ALGORITHM 15.2 #############################

## BEGIN ALGORITHM 15.3 ###########################
mutable struct EpsilonGreedyExploration
    ϵ # probability of random arm
    α # exploration decay factor
end
function (π::EpsilonGreedyExploration)(model::BanditModel)
    if rand() < π.ϵ
        π.ϵ *= π.α
        return rand(eachindex(model.B))
    else
        return argmax(mean.(model.B))
    end
end
## END ALGORITHM 15.3 #############################

## BEGIN ALGORITHM 15.4 ###########################
mutable struct ExploreThenCommitExploration
    k # pulls remaining until commitment
end
function (π::ExploreThenCommitExploration)(model::BanditModel)
    if π.k > 0
        π.k -= 1
        return rand(eachindex(model.B))
    end
    return argmax(mean.(model.B))
end
## END ALGORITHM 15.4 #############################

## BEGIN ALGORITHM 15.5 ###########################
mutable struct SoftmaxExploration
    λ # precision parameter
    α # precision factor
end
function (π::SoftmaxExploration)(model::BanditModel)
    weights = exp.(π.λ * mean.(model.B))
    π.λ *= π.α
    return rand(Categorical(normalize(weights, 1)))
end
## END ALGORITHM 15.5 #############################

## BEGIN ALGORITHM 15.6 ###########################
mutable struct QuantileExploration
    α # quantile (e.g. 0.95)
end
function (π::QuantileExploration)(model::BanditModel)
    return argmax([quantile(B, π.α) for B in model.B])
end
## END ALGORITHM 15.6 #############################

## BEGIN ALGORITHM 15.7 ###########################
mutable struct UCB1Exploration
    c # exploration constant
end
function bonus(π::UCB1Exploration, B, a)
    N = sum(b.α + b.β for b in B)
    Na = B[a].α + B[a].β
    return π.c * sqrt(log(N)/Na)
end
function (π::UCB1Exploration)(model::BanditModel)
    B = model.B
    ρ = mean.(B)
    u = ρ .+ [bonus(π, B, a) for a in eachindex(B)]
    return argmax(u)
end
## END ALGORITHM 15.7 #############################

## BEGIN ALGORITHM 15.8 ###########################
struct PosteriorSamplingExploration end

(π::PosteriorSamplingExploration)(model::BanditModel) = argmax(rand.(model.B))
## END ALGORITHM 15.8 #############################

## BEGIN ALGORITHM 15.9 ###########################
function simulate(P::MDP, model, π, h, s)
    for i in 1:h
        a = π(model, s)
        s′, r = P.TR(s, a)
        update!(model, s, a, r, s′)
        s = s′
    end
end
## END ALGORITHM 15.9 #############################

## BEGIN ALGORITHM 16.1 ###########################
mutable struct MaximumLikelihoodMDP
    S # state space (assumes 1:nstates)
    A # action space (assumes 1:nactions)
    N # transition count N(s,a,s′)
    ρ # reward sum ρ(s, a)
    γ # discount
    U # value function
    planner
end
function lookahead(model::MaximumLikelihoodMDP, s, a)
    S, U, γ = model.S, model.U, model.γ
    n = sum(model.N[s,a,:])
    if n == 0
        return 0.0
    end
    r = model.ρ[s, a] / n
    T(s,a,s′) = model.N[s,a,s′] / n
    return r + γ * sum(T(s,a,s′)*U[s′] for s′ in S)
end
function backup(model::MaximumLikelihoodMDP, U, s)
    return maximum(lookahead(model, s, a) for a in model.A)
end
function update!(model::MaximumLikelihoodMDP, s, a, r, s′)
    model.N[s,a,s′] += 1
    model.ρ[s,a] += r
    update!(model.planner, model, s, a, r, s′)
    return model
end
## END ALGORITHM 16.1 #############################

## BEGIN ALGORITHM 16.2 ###########################
function MDP(model::MaximumLikelihoodMDP)
    N, ρ, S, A, γ = model.N, model.ρ, model.S, model.A, model.γ
    T, R = similar(N), similar(ρ)
    for s in S
        for a in A
            n = sum(N[s,a,:])
            if n == 0
                T[s,a,:] .= 0.0
                R[s,a] = 0.0
            else
                T[s,a,:] = N[s,a,:] / n
                R[s,a] = ρ[s,a] / n
            end
        end
    end
    return MDP(T, R, γ)
end
## END ALGORITHM 16.2 #############################

## BEGIN ALGORITHM 16.3 ###########################
struct FullUpdate end

function update!(planner::FullUpdate, model, s, a, r, s′)
    P = MDP(model)
    U = solve(P).U
    copy!(model.U, U)
    return planner
end
## END ALGORITHM 16.3 #############################

## BEGIN ALGORITHM 16.4 ###########################
struct RandomizedUpdate
    m # number of updates
end
function update!(planner::RandomizedUpdate, model, s, a, r, s′)
    U = model.U
    U[s] = backup(model, U, s)
    for i in 1:planner.m
        s = rand(model.S)
        U[s] = backup(model, U, s)
    end
    return planner
end
## END ALGORITHM 16.4 #############################

## BEGIN ALGORITHM 16.5 ###########################
struct PrioritizedUpdate
    m # number of updates
    pq # priority queue
end
function update!(planner::PrioritizedUpdate, model, s)
    N, U, pq = model.N, model.U, planner.pq
    S, A = model.S, model.A
    u = U[s]
    U[s] = backup(model, U, s)
    for s⁻ in s
        for a⁻ in A
            n_sa = sum(N[s⁻,a⁻,s′] for s′ in S)
            if n_sa > 0
                T = N[s⁻,a⁻,s] / n_sa
                priority = T * abs(U[s] - u)
                pq[s⁻] = max(get(pq, s⁻, -Inf), priority)
            end
        end
    end
    return planner
end
function update!(planner::PrioritizedUpdate, model, s, a, r, s′)
    planner.pq[s] = Inf
    for i in 1:planner.m
        if isempty(planner.pq)
            break
        end
        update!(planner, model, dequeue!(planner.pq))
    end
    return planner
end
## END ALGORITHM 16.5 #############################

## BEGIN ALGORITHM 16.6 ###########################
function (π::EpsilonGreedyExploration)(model, s)
    A, ϵ = model.A, π.ϵ
    if rand() < ϵ
        return rand(A)
    end
    Q(s,a) = lookahead(model, s, a)
    return argmax(a->Q(s,a), A)
end
## END ALGORITHM 16.6 #############################

## BEGIN ALGORITHM 16.7 ###########################
mutable struct RmaxMDP
    S # state space (assumes 1:nstates)
    A # action space (assumes 1:nactions)
    N # transition count N(s,a,s′)
    ρ # reward sum ρ(s, a)
    γ # discount
    U # value function
    planner
    m # count threshold
    rmax # maximum reward
end
function lookahead(model::RmaxMDP, s, a)
    S, U, γ = model.S, model.U, model.γ
    n = sum(model.N[s,a,:])
    if n < model.m
        return model.rmax / (1-γ)
    end
    r = model.ρ[s, a] / n
    T(s,a,s′) = model.N[s,a,s′] / n
    return r + γ * sum(T(s,a,s′)*U[s′] for s′ in S)
end
function backup(model::RmaxMDP, U, s)
    return maximum(lookahead(model, s, a) for a in model.A)
end
function update!(model::RmaxMDP, s, a, r, s′)
    model.N[s,a,s′] += 1
    model.ρ[s,a] += r
    update!(model.planner, model, s, a, r, s′)
    return model
end
function MDP(model::RmaxMDP)
    N, ρ, S, A, γ = model.N, model.ρ, model.S, model.A, model.γ
    T, R, m, rmax = similar(N), similar(ρ), model.m, model.rmax
    for s in S
        for a in A
            n = sum(N[s,a,:])
            if n < m
                T[s,a,:] .= 0.0
                T[s,a,s] = 1.0
                R[s,a] = rmax
            else
                T[s,a,:] = N[s,a,:] / n
                R[s,a] = ρ[s,a] / n
            end
        end
    end
    return MDP(T, R, γ)
end
## END ALGORITHM 16.7 #############################

## BEGIN ALGORITHM 16.8 ###########################
mutable struct BayesianMDP
    S # state space (assumes 1:nstates)
    A # action space (assumes 1:nactions)
    D # Dirichlet distributions D[s,a]
    R # reward function as matrix (not estimated)
    γ # discount
    U # value function
    planner
end
function lookahead(model::BayesianMDP, s, a)
    S, U, γ = model.S, model.U, model.γ
    n = sum(model.D[s,a].alpha)
    if n == 0
        return 0.0
    end
    r = model.R(s,a)
    T(s,a,s′) = model.D[s,a].alpha[s′] / n
    return r + γ * sum(T(s,a,s′)*U[s′] for s′ in S)
end
function update!(model::BayesianMDP, s, a, r, s′)
    α = model.D[s,a].alpha
    α[s′] += 1
    model.D[s,a] = Dirichlet(α)
    update!(model.planner, model, s, a, r, s′)
    return model
end
## END ALGORITHM 16.8 #############################

## BEGIN ALGORITHM 16.9 ###########################
struct PosteriorSamplingUpdate end

function Base.rand(model::BayesianMDP)
    S, A = model.S, model.A
    T = zeros(length(S), length(A), length(S))
    for s in S
        for a in A
            T[s,a,:] = rand(model.D[s,a])
        end
    end
    return MDP(T, model.R, model.γ)
end
function update!(planner::PosteriorSamplingUpdate, model, s, a, r, s′)
    P = rand(model)
    U = solve(P).U
    copy!(model.U, U)
end
## END ALGORITHM 16.9 #############################

## BEGIN ALGORITHM 17.1 ###########################
mutable struct IncrementalEstimate
    μ # mean estimate
    α # learning rate function
    m # number of updates
end
function update!(model::IncrementalEstimate, x)
    model.m += 1
    model.μ += model.α(model.m) * (x - model.μ)
    return model
end
## END ALGORITHM 17.1 #############################

## BEGIN ALGORITHM 17.2 ###########################
mutable struct QLearning
    S # state space (assumes 1:nstates)
    A # action space (assumes 1:nactions)
    γ # discount
    Q # action value function
    α # learning rate
end

lookahead(model::QLearning, s, a) = model.Q[s,a]

function update!(model::QLearning, s, a, r, s′)
    γ, Q, α = model.γ, model.Q, model.α
    Q[s,a] += α*(r + γ*maximum(Q[s′,:]) - Q[s,a])
    return model
end
## END ALGORITHM 17.2 #############################

## BEGIN ALGORITHM 17.3 ###########################
mutable struct Sarsa
    S # state space (assumes 1:nstates)
    A # action space (assumes 1:nactions)
    γ # discount
    Q # action value function
    α # learning rate
    ℓ # most recent experience tuple (s,a,r)
end

lookahead(model::Sarsa, s, a) = model.Q[s,a]

function update!(model::Sarsa, s, a, r, s′)
    if model.ℓ != nothing
        γ, Q, α, ℓ = model.γ, model.Q, model.α, model.ℓ
        model.Q[ℓ.s,ℓ.a] += α*(ℓ.r + γ*Q[s,a] - Q[ℓ.s,ℓ.a])
    end
    model.ℓ = (s=s, a=a, r=r)
    return model
end
## END ALGORITHM 17.3 #############################

## BEGIN ALGORITHM 17.4 ###########################
mutable struct SarsaLambda
    S # state space (assumes 1:nstates)
    A # action space (assumes 1:nactions)
    γ # discount
    Q # action value function
    N # trace
    α # learning rate
    λ # trace decay rate
    ℓ # most recent experience tuple (s,a,r)
end

lookahead(model::SarsaLambda, s, a) = model.Q[s,a]

function update!(model::SarsaLambda, s, a, r, s′)
    if model.ℓ != nothing
        γ, λ, Q, α, ℓ = model.γ, model.λ, model.Q, model.α, model.ℓ
        model.N[ℓ.s,ℓ.a] += 1
        δ = ℓ.r + γ*Q[s,a] - Q[ℓ.s,ℓ.a]
        for s in model.S
            for a in model.A
                model.Q[s,a] += α*δ*model.N[s,a]
                model.N[s,a] *= γ*λ
            end
        end
    else
        model.N[:,:] .= 0.0
    end
    model.ℓ = (s=s, a=a, r=r)
    return model
end
## END ALGORITHM 17.4 #############################

## BEGIN ALGORITHM 17.5 ###########################
struct GradientQLearning
    A # action space (assumes 1:nactions)
    γ # discount
    Q # parameterized action value function Q(θ,s,a)
    ∇Q # gradient of action value function
    θ # action value function parameter
    α # learning rate
end
    
function lookahead(model::GradientQLearning, s, a)
    return model.Q(model.θ, s,a)
end

function update!(model::GradientQLearning, s, a, r, s′)
    A, γ, Q, θ, α = model.A, model.γ, model.Q, model.θ, model.α
    u = maximum(Q(θ,s′,a′) for a′ in A)
    Δ = (r + γ*u - Q(θ,s,a))*model.∇Q(θ,s,a)
    θ[:] += α*scale_gradient(Δ, 1)
    return model
end
## END ALGORITHM 17.5 #############################

## BEGIN ALGORITHM 17.6 ###########################
struct ReplayGradientQLearning
    A # action space (assumes 1:nactions)
    γ # discount
    Q # parameterized action value funciton Q(θ,s,a)
    ∇Q # gradient of action value function
    θ # action value function parameter
    α # learning rate
    buffer # circular memory buffer
    m # number of steps between gradient updates
    m_grad # batch size
end
    
function lookahead(model::ReplayGradientQLearning, s, a)
    return model.Q(model.θ, s,a)
end

function update!(model::ReplayGradientQLearning, s, a, r, s′)
    A, γ, Q, θ, α = model.A, model.γ, model.Q, model.θ, model.α
    buffer, m, m_grad = model.buffer, model.m, model.m_grad
    if isfull(buffer)
        U(s) = maximum(Q(θ,s,a) for a in A)
        ∇Q(s,a,r,s′) = (r + γ*U(s′) - Q(θ,s,a))*model.∇Q(θ,s,a)
        Δ = mean(∇Q(s,a,r,s′) for (s,a,r,s′) in rand(buffer, m_grad))
        θ[:] += α*scale_gradient(Δ, 1)
        for i in 1:m # discard oldest experiences
            popfirst!(buffer)
        end
    else
        push!(buffer, (s,a,r,s′))
    end
    return model
end
## END ALGORITHM 17.6 #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################

## BEGIN ALGORITHM  ###########################
## END ALGORITHM  #############################
