
```
This is how I imagine all of the functionality will integrate together into a single executable code

1. Add the path for a yearly data folder

    yearlyData = load("blah blah blah")

2. Define PlayerTag lookup table which assigns each player with an integer tag. Then, throughout the code we can just keep track of the integer tags intstead of strings with their names 

    PlayerTag = Dict("Tom Brady" => 203, "Jared Goff" => 204, etc.)

3. Main Function will iterate through each week of the season, define the S,A,R,S', do the rollout and then update Q function

````
    function preseasonRanks(weeklydata)
        #This function assigns integer "tags" to each of the 24 players (8 of ea position) used in a season.
        
    end

    function Main()

        #initialize Q-table if it doesn't already exist.

        for Game in yearlyData #run through every weekly game in a season at a time

            #Define State (output: array (or set or list or whatever) with the ranks of your three players)
            #Either a random lineup or from previous week
            State = main()

            #Define Action (need to implement exploration strategy) 
            if Game < 5
                Action = rand(1:7)
            else
                #greedy action
                Action = argmax(Q)
            end

            #Transition State: accounts for the change in the roster based on the action but is based on current rankings. 
            #Will be used as input to the rollout function to determine new state.


            #Rollout


            #Next State


            #Reward


            return State, Action, Reward, NextState
        end
    end

    function QLearning(state,action,reward,next_state)
    end