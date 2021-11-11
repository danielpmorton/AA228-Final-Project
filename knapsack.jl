# Algorithm 19.7 in Algorithms for Optimization
# A method for solving the 0-1 knapsack problem with:
# Item values v
# Integral item weights w
# Integral capacity w_max
# Note: Recovering the design vector from the cached solution requires additional iteration

function knapsack(v, w, w_max)
    n = length(v)
    y = Dict((0,j) => 0.0 for j in 0:w_max)
    for i in 1 : n
        for j in 0 : w_max
            y[i,j] = w[i] > j ? y[i-1,j] : max(y[i-1,j], y[i-1,j-w[i]] + v[i])
        end
    end
    
    # recover solution
    x, j = falses(n), w_max
    for i in n: -1 : 1
        if w[i] ≤ j && y[i,j] - y[i-1, j-w[i]] == v[i]
            # the ith element is in the knapsack
            x[i] = true
            j -= w[i]
        end
    end
    return x
end