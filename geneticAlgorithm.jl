using Distributions


# Algorithm 9.4 in Algorithms for Optimization
# Takes:
    # Objective function f
    # Initial population
    # Number of iterations k_max
    # SelectionMethod S
    # CrossoverMethod c
    # MutationMethod M

function genetic_algorithm(f, population, k_max, S, C, M)
    for k in 1 : k_max
        parents = select(S, f.(population))
        children = [crossover(C,population[p[1]],population[p[2]]) for p in parents]
        population .= mutate.(Ref(M), children)
    end
    population[argmin(f.(population))]
end


# Other algorithms from Chapter 9 in Algorithms for Optimization

## POPULATION INITIALIZATION METHODS ##

function rand_population_uniform(m, a, b)
    d = length(a)
    return [a+rand(d).*(b-a) for i in 1:m]
end

function rand_population_normal(m, μ, Σ)
    D = MvNormal(μ,Σ)
    return [rand(D) for i in 1:m]
end

function rand_population_cauchy(m, μ, σ)
    n = length(μ)
    return [[rand(Cauchy(μ[j],σ[j])) for j in 1:n] for i in 1:m]
end

rand_population_binary(m, n) = [bitrand(n) for i in 1:m]

## SELECTION METHODS ##

abstract type SelectionMethod end

struct TruncationSelection <: SelectionMethod
    k # top k to keep
end

function select(t::TruncationSelection, y)
    p = sortperm(y)
    return [p[rand(1:t.k, 2)] for i in y]
end

struct TournamentSelection <: SelectionMethod
    k
end

function select(t::TournamentSelection, y)
    getparent() = begin
        p = randperm(length(y))
        p[argmin(y[p[1:t.k]])]
    end
    return [[getparent(), getparent()] for i in y]
end

struct RouletteWheelSelection <: SelectionMethod end

function select(::RouletteWheelSelection, y)
    y = maximum(y) .- y
    cat = Categorical(normalize(y, 1))
    return [rand(cat, 2) for i in y]
end

## CROSSOVER METHODS ##

abstract type CrossoverMethod end

struct SinglePointCrossover <: CrossoverMethod end

function crossover(::SinglePointCrossover, a, b)
    i = rand(1:length(a))
    return vcat(a[1:i], b[i+1:end])
end

struct TwoPointCrossover <: CrossoverMethod end

function crossover(::TwoPointCrossover, a, b)
    n = length(a)
    i, j = rand(1:n, 2)
    if i > j
        (i,j) = (j,i)
    end
    return vcat(a[1:i], b[i+1:j], a[j+1:n])
end

struct UniformCrossover <: CrossoverMethod end

function crossover(::UniformCrossover, a, b)
    child = copy(a)
    for i in 1 : length(a)
        if rand() < 0.5
            child[i] = b[i]
        end
    end
    return child
end

struct InterpolationCrossover <: CrossoverMethod
    λ
end

crossover(C::InterpolationCrossover, a, b) = (1-C.λ)*a + C.λ*b

## MUTATION METHODS ##

abstract type MutationMethod end

struct BitwiseMutation <: MutationMethod
    λ
end

function mutate(M::BitwiseMutation, child)
    return [rand() < M.λ ? !v : v for v in child]
end

struct GaussianMutation <: MutationMethod
    σ
end

function mutate(M::GaussianMutation, child)
    return child + randn(length(child))*M.σ
end