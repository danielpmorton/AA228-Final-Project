# Script to test a policy

using DataFrames
using CSV
using Statistics
using Random
using Printf 

include("stateJustRank.jl") 
include("IntegratedFunctions_QLearning.jl") 


function main_eval()
    # Initializations and parameters
    policyFileName = "TestRun_Policy.csv"
    numStates = 512
    numActions = 7
    YearFileLocation = "newweekly/2017"
    CumulativeReward = []
    Rank = Dict{String, Int64}() # Initializing as empty dictionary with defined key => value types
    global NumWeeks = size(readdir(YearFileLocation),1)
    QB_Players, RB_Players, WR_Players = PlayerTags(YearFileLocation)
    compareFileName = "compare.csv"
    State = []
    CurrentStateLineup = Dict[]

    # Loading the policies for test
    ourPolicy = loadPolicy(policyFileName)
    randomPolicy = makeRandomPolicy(numStates, numActions)

    # Initialize the first state
    CurrentStateLineup = makeRandomLineup(QB_Players, RB_Players, WR_Players)
    currentWeekData = Rollout(YearFileLocation, 1, QB_Players, RB_Players, WR_Players)
    Rank = getPlayerRankings(CurrentStateLineup, currentWeekData)
    State = makeState(Rank)   

    # Evaluate our policy
    rewards_ourPolicy = evaluatePolicy(ourPolicy, State)
    rewards_randomPolicy = evaluatePolicy(randomPolicy, State)

    # Goal: plot the cumulative reward over the course of the season for random and our action
    compareDF = DataFrame(ours = rewards_ourPolicy, random= rewards_randomPolicy)
    CSV.write(compareFileName, compareDF)

    # return rewards_ourPolicy, rewards_randomPolicy
end

function evaluatePolicy(policy, State)
    # Pass in the policy to evaluate as well as the initial state

    CumulativeReward = []

    for i in 2:NumWeeks 

        Action = policy[State]
        preRolloutInt = i - 1
    
        TransitionRank = Transition(Rank,Action)
        prevWeekData = Rollout(YearFileLocation, preRolloutInt, QB_Players, RB_Players, WR_Players)
        #Keep track of the new lineup we want, this will be converted to our new state after the rollout once we have new rankings 
        NextStateLineup = RankingsToPlayers(TransitionRank,prevWeekData)
    
        RolloutTable = Rollout(YearFileLocation, i, QB_Players, RB_Players, WR_Players)
    
        NextStateRank = getPlayerRankings(NextStateLineup, RolloutTable)
        NextState = makeState(NextStateRank)
    
        Reward = CalculateReward(NextStateLineup, NextStateRank, Action, RolloutTable, CurrentStateLineup)
        push!(CumulativeReward, Reward)
    
        #Append the data text file with this iteration's information
        #The ranks are dictionaries so need to convert to arrays for the output file
        RankArray = [Rank["QB"],Rank["RB"],Rank["WR"]]
        NextStateRankArray = [NextStateRank["QB"],NextStateRank["RB"],NextStateRank["WR"]]
    
        #Update State and Rank for next iteration
        State = copy(NextState)
        CurrentStateLineup = copy(NextStateLineup)
        Rank = copy(NextStateRank)
    
    end

    return CumulativeReward
end


function loadPolicy(policyFileName)
    # Load the file
    mydata = CSV.read(policyFileName, DataFrame)
    # Make array
    mat = Matrix(mydata)
    policy = mat[:,2]
    return policy
end

function makeRandomPolicy(numStates, numActions)
    # Generate an array of random actions of length numStates
    policy  = rand(1:numActions, numStates)
    return policy
end


#######
main_eval()

