```
UPDATE **************
- reducing the number of players per position to 8
- reducing the definition of mean and variance to just "above" vs "below" average

CODE OVERVIEW:

Main question: "How do we define the state of our Q-learning model"

Goals:
- Read from the tables generated in the CSV-reading/sorting code
- In the state, represent the current ranks of the players, as well as info on their 
  historical performance (mean and variance in points earned)
- Get the state into a scalar value so it is easy to index into the Q table

Final output:
state -- a number which encodes the ranking of the QB, RB, WR
      (from the makeState function)

Some notes:
- As of 11/24: needs to be integrated better with Walter's code!
  -- note the structure of the data passed into getPlayerRankings() !!
```

using DataFrames
using CSV # Check if actually using this ones
using Statistics
using Random

# NOTE: remember to update these with the reduced 8-person/position league
# Define global variables
global positions= ["QB", "RB", "WR"]
global QBs = ["Jared Goff","Tom Brady","Dak Prescott","Deshaun Watson","Philip Rivers","Russell Wilson","Aaron Rodgers","Kirk Cousins"]
global RBs = ["Ezekiel Elliott","Nick Chubb","Christian McCaffrey","Dalvin Cook","Chris Carson","Leonard Fournette","Mark Ingram","Todd Gurley"]
global WRs = ["Michael Thomas","Keenan Allen","DeAndre Hopkins","Julio Jones","Allen Robinson","Tyler Lockett","Stefon Diggs","Chris Godwin"]


# Build the main function here, run it at the very end of the file
function main()
    lineup = namesToLineup("Tom Brady", "Christian McCaffrey", "Chris Godwin") # OR: lineup = makeRandomLineup(QBs, RBs, WRs)
    ranks = getPlayerRankings(lineup, currentWeekData)
    state = makeState(ranks)
    print(state)
end

##### Begin sub functions #####

function namesToLineup(QB_name, RB_name, WR_name)
    ``` Converts three player names (strings) into a dictionary mapping positions => names ```
    lineup = Dict([("QB", QB_name), ("RB", RB_name), ("WR", WR_name)])
    return lineup
end

# NOTE THE UNIQUE INPUT FORMAT IN THIS ONE
# Assumes we have a table like the others but with only one column of week data called "points" ****
# player, position, points
function getPlayerRankings(lineup, currentWeekData)
    ```
    Description: Takes the most recent weekly data, ranks the players, and assigns a rank to the players in your lineup
    Inputs:
    Outputs: Dictionary mapping the positions => rank for the chosen player
    ```
    # Separate the data by position
    QBdata = currentWeekData[currentWeekData.position .== "QB", :]
    RBdata = currentWeekData[currentWeekData.position .== "RB", :]
    WRdata = currentWeekData[currentWeekData.position .== "WR", :]
    # Sort the data by points from highest to lowest, so the first player has the highest score
    sort!(QBdata, [:points], rev=true) 
    sort!(RBdata, [:points], rev=true) 
    sort!(WRdata, [:points], rev=true) 
    # Get the rank based on the row index corresponding to your player
    QBrank = findfirst(QBdata.player .== lineup["QB"])
    RBrank = findfirst(RBdata.player .== lineup["RB"])
    WRrank = findfirst(WRdata.player .== lineup["WR"])
    # Output this as a dictionary storing the position => rank
    ranks = Dict([("QB", QBrank), ("RB", RBrank), ("WR", WRrank)])
    return ranks
end

function makeRandomLineup(QBs, RBs, WRs)
    ```
    Description: Creates a random lineup for the week
    Inputs: Arrays storing the names of all QBs, RBs, and WRs being considered in the league
    Outputs: A dictionary storing the position and the associated name of the player -- Example: "QB" => "Tom Brady"
    ```
    QB_name = QBs[rand((1:length(QBs)))]
    RB_name = RBs[rand((1:length(RBs)))]
    WR_name = WRs[rand((1:length(WRs)))]
    lineup = namesToLineup(QB_name, RB_name, WR_name)
    return lineup
end

function makeState(ranks)
    ```
    Description: Get an integer value based on the rank and the comparison 
    Inputs: ranks: Dictionary mapping "position" => rank in the league
    Outputs: state -- a number which encodes the ranking of the QB, RB, WR 
                   -> uses binary encoding and then converts to integer with with range (0,511)
    ```
    # Combine our info together into a 15-bit binary string
    # First, convert the ranks for each player into a binary number
    QBrank_binary = digits((ranks["QB"]-1), base=2, pad=3) |> reverse
    RBrank_binary = digits((ranks["RB"]-1), base=2, pad=3) |> reverse
    WRrank_binary = digits((ranks["WR"]-1), base=2, pad=3) |> reverse
    # Now, combine these all together in a string
    # Note: the performance stats are 0 or 1 now and don't need to be converted to binary
    bitstring = string(QBrank_binary) * string(RBrank_binary) * string(WRrank_binary)
    # Un-binarify this value
    state = parse(Int, bitstring, base=2)
    return state
end

##### RUN THE MAIN FUNCTION #####
main() 
