
#=
OVERVIEW:

This function evaluates a learned policy and random policy for the same initial state. 

=#

using DataFrames
using CSV
using Statistics
using Random
using Printf 

include("stateJustRank.jl") 

function RunAllYears()

    for Year in 1999:2017
        GatherData(Year)
    end
    
end

function GatherData(Year)

    Iterations = 20 
    YearFileLocation = "dirtyweekly/" * string(Year) 
    OutputFileName = "EvaluatedData/" * string(Year)

    OurrewardStorageFileName = OutputFileName * "_QLearned"
    RandomrewardStorageFileName = OutputFileName * "_Random"

    Qpolicy = loadPolicy("TestRun_Policy.csv")
    RandomPolicy = makeRandomPolicy(512,7)

    OurRewardTable = Array{Any}(undef, Iterations, 16)
    RandomRewardTable = Array{Any}(undef, Iterations, 16)

    #Iterations is how many times to run each season 
    for i = 1:Iterations 

        #Initialize State properties so that we can compare our learned policy to a random policy from same initial state
        QB_Players, RB_Players, WR_Players = PlayerTags(YearFileLocation)
        CurrentStateLineup = makeRandomLineup(QB_Players, RB_Players, WR_Players)
        currentWeekData = Rollout(YearFileLocation, 1, QB_Players, RB_Players, WR_Players)
        Rank = getPlayerRankings(CurrentStateLineup, currentWeekData)
        initialState = makeState(Rank)

        OurCumulativeReward = EvaluatePolicy(YearFileLocation, Qpolicy,Rank, initialState, CurrentStateLineup, currentWeekData, QB_Players, RB_Players,WR_Players)
        #println("Random Policy")
        RandomCumulativeReward = EvaluatePolicy(YearFileLocation, RandomPolicy, Rank, initialState, CurrentStateLineup, currentWeekData, QB_Players, RB_Players,WR_Players)

        OurRewardTable[i,:] = transpose(OurCumulativeReward)
        RandomRewardTable[i,:] = transpose(RandomCumulativeReward)

    end 

    CSV.write(OurrewardStorageFileName, DataFrame(OurRewardTable,:auto))
    CSV.write(RandomrewardStorageFileName, DataFrame(RandomRewardTable,:auto))
    
end 

function mainPolicyEval()
    
    YearFileLocation = "dirtyweekly/2019"
    
    QOutputFileName = "QtestEvalPolicy"
    Qpolicy = loadPolicy("TestRun_Policy.csv")

    RandomOutFileName = "RandomEvalPolicy"
    RandomPolicy = makeRandomPolicy(512,7)

    #Initialize State properties so that we can compare our learned policy to a random policy from same initial state
    QB_Players, RB_Players, WR_Players = PlayerTags(YearFileLocation)
    CurrentStateLineup = makeRandomLineup(QB_Players, RB_Players, WR_Players)
    currentWeekData = Rollout(YearFileLocation, 1, QB_Players, RB_Players, WR_Players)
    Rank = getPlayerRankings(CurrentStateLineup, currentWeekData)
    initialState = makeState(Rank)

    #Evalue learned policy and random policy from same initial state 
    #println("Learned Policy")
    OurReward = EvaluatePolicy(YearFileLocation, Qpolicy,Rank, initialState, CurrentStateLineup, currentWeekData, QB_Players, RB_Players,WR_Players)
    #println("Random Policy")
    RandomReward = EvaluatePolicy(YearFileLocation, RandomPolicy, Rank, initialState, CurrentStateLineup, currentWeekData, QB_Players, RB_Players,WR_Players)
 
end
    

function EvaluatePolicy(YearFileLocation, policy, Rank, State, CurrentStateLineup, currentWeekData, QB_Players, RB_Players, WR_Players)
    #Number of iterations for season
    NumWeeks = size(readdir(YearFileLocation),1)

    #rewardStorageFileName = OutputFileName * "_" * "Rewards" * ".csv"
    CumulativeReward = []
   
            
        # Season for-loop
        #Run through every weekly game in a season at a time. Start at week 2 because week 1 is reserved for initializing the first state.
        for i in 2:NumWeeks 
            
            Action = SelectAction(policy,State,Rank)
            
            ######## STEP 4: CALCULATE TRANSITION STATE #################
            #Transition State: accounts for the change in the roster based on the action but is based on current rankings. 
            #If we are on the first iteration (i.e. no rollouts have happened, just use the first week's data) Otherwise, use the data of the previous iteration)
            
            preRolloutInt = i - 1
            TransitionRank = Transition(Rank,Action)
            prevWeekData = Rollout(YearFileLocation, preRolloutInt, QB_Players, RB_Players, WR_Players)
            NextStateLineup = RankingsToPlayers(TransitionRank,prevWeekData)

            ######## STEP 5: ROLLOUT ################
            #Rollout Function Input/Output 
            #Input: Year, Week, Dictionaries of players for each position 
            #Output: Table with Player Name, Player ID, Position, and Fantasy Points 
            #Daniel's function "getPlayerRankings" does this already but need to make sure that the input to that function is a dataset with only the 24 players we are dealing with or else 
            RolloutTable = Rollout(YearFileLocation, i, QB_Players, RB_Players, WR_Players)

            ######### STEP 6: CALCULATE NEXT STATE ################
            #Recalculate the state based on new rankings, Daniel's functions already do this:
            NextStateRank = getPlayerRankings(NextStateLineup, RolloutTable)
            NextState = makeState(NextStateRank)

            ########## STEP 7: CALCULATE REWARD ##############
            #CumulativeReward will be an array with the reward for every week's lineup. We'll keep track of this to show our agent improving over time. 
            Reward = CalculateReward(NextStateLineup, NextStateRank, Action, RolloutTable, CurrentStateLineup)
            push!(CumulativeReward, Reward)

            #Debugging - Print each week
            # println(" ")
            # println("Game: ", i)
            # println("State is:", Rank)
            # println("Action is: ", Action)
            # println("Transition State: ", TransitionRank)
            # println("New State is: ", NextStateRank)

            #Update State and Rank for next iteration
            State = copy(NextState)
            CurrentStateLineup = copy(NextStateLineup)
            Rank = copy(NextStateRank)

        end

    return CumulativeReward
    #CSV.write(rewardStorageFileName, DataFrame(reward = CumulativeReward))

end

###### SUBFUNCTIONS ##################

function PlayerTags(YearFileLocation)
    #This function randomly selects 8 players from each position and assigns integer "tags" to each of them
    #Input: Path of location of folder with csv files for a yearly season 
    #Output: QB_Players, RB_Players, WR_Players, dictionaries that maps players to integers for each position. 

    #Step 1: Load in CSV of Week 1 for the input season as a dataframe
    Week1Path = YearFileLocation * "/week1.csv"
    Week1Data = CSV.read(Week1Path, DataFrame)
    
    #Using Walter's code from rankandrollout to create arrays for each position
    QBlist = []
    RBlist = []
    WRlist = []

    # Walter's code: get list of player names for each position to use for sorting
    for i = 1:size(Week1Data,1)
        if Week1Data.Pos[i] == "QB"
            push!(QBlist,Week1Data.Player[i])
        elseif Week1Data.Pos[i]== "RB"
            push!(RBlist,Week1Data.Player[i])
        elseif Week1Data.Pos[i] == "WR"
            push!(WRlist,Week1Data.Player[i])
        end
    end

    # Walter's code: pick random subset of 10 players at each position
    shuffleQBlist = randcycle(length(QBlist))
    shuffleRBlist = randcycle(length(RBlist))
    shuffleWRlist = randcycle(length(WRlist))

    QBsubset = QBlist[shuffleQBlist[1:8]]
    RBsubset = RBlist[shuffleRBlist[1:8]]
    WRsubset = WRlist[shuffleWRlist[1:8]]

    #Create Dictionaries for each position with an integer key and player name value
    QB_Players = Dict(i => QBsubset[i] for i=1:size(QBsubset,1))
    RB_Players = Dict(i => RBsubset[i] for i=1:size(RBsubset,1))
    WR_Players = Dict(i => WRsubset[i] for i=1:size(WRsubset,1))

    QB_Players, RB_Players, WR_Players
end

function SelectAction(policy, state, rank)
    #This function selects an action using an epsilon-greedy exploration strategy. Not all actions are possible from each state so the function checks to see which actions are possible and then selects.
    #Inputs: State, Rank - dictionary of current state, Q - action value matrix for selecting greedy action, and epsilon - learning rate 
    #Output: Single integer from 1-7 that represents action taken

    #convert rank dictionary to an array so it's easier to work with
    rankArray = [rank["QB"], rank["RB"], rank["WR"]]

    #Check available actions. Note, actions are:
        #Action 1: Swap QB Up
        #Action 2: Swap QB down
        #Action 3: Swap RB Up
        #Action 4: Swap RB down
        #Action 5: Swap WR Up
        #Action 6: Swap WR down
        #Action 7: Do Nothing 
        Action = 0
        i = 1
        
        while Action == 0
            Action = policy[state]

            #Check violations 
            if Action == 1 && rankArray[1] == 1
                Action = 0
            elseif Action == 2 && rankArray[1] == 8
                Action = 0
            elseif Action == 3 && rankArray[2] == 1
                Action = 0
            elseif Action == 4 && rankArray[2] == 8
                Action = 0
            elseif Action == 5 && rankArray[3] == 1
                Action = 0
            elseif Action == 6 && rankArray[3] == 8
                Action = 0
            end
        
            i += 1
        
            #if it's looping for too long, just choose "Do Nothing" action 
            if i == 20
                Action = 7
            end
        
        end
        Action 

end
    

function RankingsToPlayers(LineupRank, weeklydata)
    #the reverse function of Daniel's "getPlayerRankings", this function will map the rankings in our state back to player names)
    #Input: Rankings of your lineup, weeklydata in the form generated by the RolloutFunction
    #Output: A dictionary with postion mapped to player name 

    # Separate the data by position
    QBdata = weeklydata[weeklydata.position .== "QB", :]
    RBdata = weeklydata[weeklydata.position .== "RB", :]
    WRdata = weeklydata[weeklydata.position .== "WR", :]

    # Sort the data by points from highest to lowest, so the first player has the highest score
    sort!(QBdata, [:points], rev=true) 
    sort!(RBdata, [:points], rev=true) 
    sort!(WRdata, [:points], rev=true) 

    QBName = QBdata.player[LineupRank["QB"]]
    RBName = RBdata.player[LineupRank["RB"]]
    WRName = WRdata.player[LineupRank["WR"]]

    Lineup = Dict("QB" => QBName, "RB" => RBName, "WR" => WRName)
    
end     

function Rollout(YearFileLocation, weekNum, QB_Players, RB_Players, WR_Players)
    #This function formats the data for a specified week and returns a dataframe with only the subset of players that are being used in the season. 
    #This function in conjunction with Daniel's "getPlayerRankings" can be used to determine our new state.
    #Input: Location of the season we are using, the week number, and the dictionaries of players we are using in the season 
    #Output: DataFrame with player, position, and points for the 24 players (8 of each position) we are using. 


    FilePath = YearFileLocation * "/week" * string(weekNum) * ".csv"

    #Step 1: Load all data into dataframe
    rawData = CSV.read(FilePath, DataFrame)

    #Step 2: We only need Player, Position, and Points columns
    limitedData = DataFrame(player = rawData.Player, position = rawData.Pos, points = rawData.StandardFantasyPoints)

    #Step 3: We only care about the subset of players in QB_Players, RB_Players and WR_Players 
    #merge player dictionaries into 1 dictionary with all players and tags
    Players = [QB_Players,RB_Players,WR_Players]

    #initialize data DataFrame
    RolloutTable = DataFrame(player = Any[], ID = Any[], position = Any[], points = Any[])

    #this is so inefficient but creating final dataframe with just the subset of players we are using and adding the player tag
    for j in 1:length(Players)
        for i in 1:length(Players[j])
            row = findfirst(limitedData.player .== Players[j][i])
            dataArray = [limitedData.player[row], i, limitedData.position[row], limitedData.points[row]]
            push!(RolloutTable, dataArray)
        end
    end
    RolloutTable
end

function Transition(Rank, Action)
    #implements actions to update the state tuple before we run the rollout, outputs ranks 

    transitionRank = copy(Rank)

    if Action == 1 # swap QB up
            new = transitionRank["QB"] - 1
            delete!(transitionRank, "QB")
            transitionRank["QB"] = new 

    elseif Action == 2 # swap QB down
            new = transitionRank["QB"] + 1
            delete!(transitionRank, "QB")
            transitionRank["QB"] = new 

    elseif Action == 3 # swap RB up
            new = transitionRank["RB"] - 1
            delete!(transitionRank, "RB")
            transitionRank["RB"] = new 

    elseif Action == 4 # swap RB down
            new = transitionRank["RB"] + 1
            delete!(transitionRank, "RB")
            transitionRank["RB"] = new

    elseif Action == 5 # swap WR up
            new = transitionRank["WR"] - 1
            delete!(transitionRank, "WR")
            transitionRank["WR"] = new

    elseif Action == 6 # swap WR down
            new = transitionRank["WR"] + 1
            delete!(transitionRank, "WR")
            transitionRank["WR"] = new

    end

    transitionRank
    
end

function CalculateReward(NextStateLineup, NextStateRank, Action, RolloutTable, CurrentStateLineup)
    #The reward for each iteration will be composed of a transaction cost for trading a player plus the fantasy points scored by the lineup

    #Hyperparameters: We can vary how much the transaction cost of trading a specific position should be scaled by
    #General_Scalar is so that transaction cost is same order of magnitude as fantasy points
    #Total Transaction Cost = (Position_Transaction)* (General_Scalar) * (Inverse of New Rank)
    #Currently, trading down gives you a positive reward, not sure if we want to change this
    QB_Transaction = 5
    RB_Transaction = 1.5
    WR_Transaction = 1
    General_Scalar = 10

    if Action == 1 # swap QB up
        Transaction_Cost = -1 * 1/Int(NextStateRank["QB"])
        Scaled_Cost = General_Scalar * QB_Transaction * Transaction_Cost  

    elseif Action == 2 # swap QB down
        Transaction_Cost = 1/Int(NextStateRank["QB"])
        Scaled_Cost = General_Scalar * QB_Transaction * Transaction_Cost 

    elseif Action == 3 # swap RB up
            Transaction_Cost = -1* 1/Int(NextStateRank["RB"])
            Scaled_Cost = General_Scalar * RB_Transaction * Transaction_Cost 

    elseif Action == 4 # swap RB down
            Transaction_Cost = 1/Int(NextStateRank["RB"])
            Scaled_Cost = General_Scalar * RB_Transaction * Transaction_Cost 

    elseif Action == 5 # swap WR up
            Transaction_Cost = -1* 1/Int(NextStateRank["WR"])
            Scaled_Cost = General_Scalar * WR_Transaction * Transaction_Cost 

    elseif Action == 6 # swap WR down
            Transaction_Cost = 1/Int(NextStateRank["WR"])
            Scaled_Cost = General_Scalar * WR_Transaction * Transaction_Cost 

    elseif Action == 7 #do nothing
            Transaction_Cost = 0

    end

    #Fantasy Points for each player
    QBrow = findfirst(RolloutTable.player .== NextStateLineup["QB"])
    RBrow = findfirst(RolloutTable.player .== NextStateLineup["RB"])
    WRrow = findfirst(RolloutTable.player .== NextStateLineup["WR"])

    QBrow_old =findfirst(RolloutTable.player .== CurrentStateLineup["QB"])
    RBrow_old = findfirst(RolloutTable.player .== CurrentStateLineup["RB"])
    WRrow_old = findfirst(RolloutTable.player .== CurrentStateLineup["WR"])

    NewTotalFantasyPoints = RolloutTable.points[QBrow] + RolloutTable.points[RBrow] + RolloutTable.points[WRrow]
    OldTotalFantasyPoints = RolloutTable.points[QBrow_old] + RolloutTable.points[RBrow_old] + RolloutTable.points[WRrow_old]

    TotalReward = Transaction_Cost + NewTotalFantasyPoints - OldTotalFantasyPoints

end

function QLearning(Q,state,action,reward,next_state)
    gamma = 0.95
    alpha = 0.5
    
    Q[state,action] += alpha*(reward + gamma*maximum(Q[next_state,:]) - Q[state,action])
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
#################
# @time main()
# main()
