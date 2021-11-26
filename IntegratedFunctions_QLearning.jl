
#=
OVERVIEW:

There will be a Main function that will loop through every week of a season and define the inputs for Q-learning as well as update the Q-table.

    Current Subfunctions that Zahra needs to fill out:
    -PlayerTags - DONE
    -RankingstoPlayers
    -Rollout
    -TransitionState
    -CalculateReward 
    -QLearning - DONE

    Current Issues that still need to be addressed:
    -How are we accounting for running multiple seasons or multiple initial lineups that all populate the same Q-table?
    -

=#
    function IntegratedFnc(YearFileLocation)
        #This function will define all of the inputs for Q-Learning (state, action, reward, and next state). In order to define the reward and next state, we need to conduct a rollout.
        #Inputs
            # Year_FileLocation: Path of location of folder with csv files for a yearly season
            # Players: Dictionary with players for that season and their integer tag. Generated by function PlayerTags, see subfunctions below. 
        #Outputs 
            #CumulativeReward: Array of Reward for every week so that we can plot it 
            #Q : Matrix of Action Value for each state, action pair 


        #Initialize State and Q-Table to empty. NOTE: This will need to change once I figure out how to handle Q from season to season
        State = []
        StateSpace = 1000
        ActionSpace = 7
        Q = zeros(StateSpace, ActionSpace)

        #Initialize the Players that will be used for the season
        QB_Players, RB_Players, WR_Players = PlayerTags(YearFileLocation)

        #Number of weeks in a specified season, used to determine number of iterations 
        NumWeeks = size(readdir(YearFileLocation),1)

        #run through every weekly game in a season at a time
        for i in 1:NumWeeks 

            #STATE________________________________________________________________________________________
            #Define State (use Daniel's stateJustRank functions)
            #Either a random lineup or from previous week (if using from previous week, state will be defined after Q-table is updated)
            if isempty(State)
                #generate random lineup using week 1 data
                lineup = makeRandomLineup(QB_Players, RB_Players, WB_Players)
                currentWeekData = Rollout(YearFileLocation, 1, QB_Players, RB_Players, WR_Players)
                rank = getPlayerRankings(lineup, currentWeekData)
                state = makeState(rank)

                #Keep track of which Player Integers correspond to the rankings in State
                StatePlayerTags = RankingToPlayers(State) 
            end 
                
            
            #ACTION_________________________________________________________________________________________
            #NEEDS WORK: Define Action (need to implement exploration strategy) 
            if i < 5
                Action = rand(1:7)
            else
                #greedy action
                Action = argmax(Q)
            end


            #TRANSITION STATE_______________________________________________________________________________
            #Transition State: accounts for the change in the roster based on the action but is based on current rankings. 
            Transition_State = TransitionState(State,Action)
            #Keep track of which Player Integers correspond to the rankings in State
            NextStatePlayerTags = RankingToPlayers(Transition_State)

            #ROLLOUT________________________________________________________________________________________
            #Rollout Function Input/Output 
                #Input: Year, Week, Array of Player Integer Tag (see PlayerTags function)
                #Output: Table with Player Integer, Rank for the week, Position, and Fantasy Points 
                #Daniel's function "getPlayerRankings" does this already but need to make sure that the input to that function is a dataset with only the 24 players we are dealing with or else 


            #NEXT STATE_______________________________________________________________________________________
                #Recalculate the state based on new rankings, Daniel's functions already do this:
                    #getPlayerRankings (input: NextStatePlayerTags)
                    
                    NextState = makeState  

            #REWARD____________________________________________________________________________________________
                #CumulativeReward will be an array with the reward for every week's lineup. We'll keep track of this to show our agent improving over time. 

            #Q-LEARNING___________________________________________________________________________________________
            #Update Q Table
            QLearning(Q,State,Action,Reward,NextState)

            #Update State for next iteration
            State = NextState
 
        end
        return Q, CumulativeReward
    end

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


    function RankingstoPlayers(State, weeklydata)
        #the reverse function of Daniel's "getPlayerRankings", this function will map the rankings in our state back to the player tags generated by PlayerTags)
        return PlayerState
    end 

    function Rollout(YearFileLocation,weekNum, QB_Players, RB_Players, WR_Players)
    end


    function TransitionState(State, Action)
        #implements actions to update the state tuple before we run the rollout
        return TransitionState 
    end

    function CalculateReward(NextState, Action)
        
    end

    function QLearning(Q,state,action,reward,next_state)
        gamma = 0.95
        alpha = 0.5
        
        Q[state,action] += alpha*(reward + gamma*maximum(Q[next_state,:]) - Q[state,action])
    end



