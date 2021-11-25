```
CODE OVERVIEW:

Main question: "How do we define the state of our Q-learning model"

Notes:
- It's easiest if the state is a scalar value, as this can be easily indexed in a Q table
- 

Overview:
This code will... 

This code makes heavy use of dictionaries

```

using DataFrames
using CSV # Check if actually using this ones
using Statistics
using Random


# Define global variables ** NOTE: CHECK IF THIS GLOBAL SYNTAX WORKS **
global positions= ["QB", "RB", "WR"]
global QBs = ["Jared Goff","Tom Brady","Dak Prescott","Deshaun Watson","Philip Rivers","Russell Wilson","Aaron Rodgers","Kirk Cousins","Matt Ryan","Derek Carr"]
global RBs = ["Ezekiel Elliott","Nick Chubb","Christian McCaffrey","Dalvin Cook","Chris Carson","Leonard Fournette","Mark Ingram","Todd Gurley","Alvin Kamara","Aaron Jones"]
global WRs = ["Michael Thomas","Keenan Allen","DeAndre Hopkins","Julio Jones","Allen Robinson","Tyler Lockett","Stefon Diggs","Chris Godwin","Mike Evans","Jarvis Landry"]




lineup = Dict([("QB", "Tom Brady"), ("RB", "Christian McCaffrey"), ("WR", "Mike Evans")]) # FOR DEBUGGING PURPOSES




function getLeaguePerformanceData(data)
    ```
    Description:
            Step 0: Get the full history for the league: all players, all weeks
            Step 1: For each player, calculate a points_mean and points_variance
            Step 2: For each position, take the players' means and find a mean(points_mean) and variance(points_mean)
                    Also: .............................. variances and find a mean (points_variance) and variance(points_variance)
            Step 3: Store the data in dictionary: Dict["Position"] (meanOfMean, meanOfVariance, varianceOfMean, varianceofVariance)
    Inputs: League performance history over the entire *previous* season (or multiple seasons)
            Format: DataFrame. Rows are players, columns are weeks, entries are points scored *** NOTE: see walter code output formatting ***
    Outputs: Dictionary with (key, value) = ("Position", (mean, variance)) of type {String, Tuple}
    ```
    # Initialize the dictionary to store the league performance
    stats = Dict{String, Tuple}()

    for pos in positions
        # Extract all rows in the dataframe where the entry of the position column matches the current position in the loop
        positionData = data[data.position .== pos, :] 
        # Calculate the average over all weeks # NOTE: may want to exclude 0s in the data - didn't play that week?
        # Extract just the points info from the table and load to matrix (easier to work with )
        pointsOnly = Matrix(positionData[:,3:end]) # NOTE: this depends on the indexing in the dataframe!! Currently assumes columns 1,2 are player, position

        # Calculate the mean and variance for each player
        meanPerPlayer = mean(pointsOnly, dims=2)  # DF Form: numWeeks = size(pointsOnly)[2]   ;   meanPerPlayer = sum(eachcol(pointsOnly))/numWeeks
        varPerPlayer = var(pointsOnly, dims=2)

        # Calculate the overall means and variances for a position
        meanOfMean = mean(meanPerPlayer)
        meanOfVar = mean(varPerPlayer)
        varOfMean = var(meanPerPlayer)
        varOfVar = var(varPerPlayer)

        stats[pos] = (meanOfMean, meanOfVar, varOfMean, varOfVar)
    end
    return stats
end

function getLineupPerformanceData(lineup, data)
    ```
    Description:
    Inputs:
    Outputs:
    ```
    # lineup will be an array/vector containing the names of the three players in order QB, RB, WR
    # Using array/vector instead of set since sets can be unordered
    # output will be the mean and variance of the players
    stats = Dict{String, Tuple}()
    for (pos, name) in lineup
        # Extract all row in the dataframe corresponding to the player of choice
        playerData = data[data.player .== name, :] 
        # Calculate the average over all weeks # NOTE: may want to exclude 0s in the data - didn't play that week?
        # Extract just the points info from the table and load to matrix (easier to work with )
        pointsOnly = Matrix(playerData[:,3:end]) # NOTE: this depends on the indexing in the dataframe!! Currently assumes columns 1,2 are player, position

        # Calculate the mean and variance
        mean = mean(pointsOnly, dims=2) 
        var = var(pointsOnly, dims=2)

        # Assign to dictionary based on the *position* not the player name
        # Because dictionaries are unordered and this makes it easier to directly compare stats per position
        stats[pos] = (mean, var)
    end

    return stats
end


function evaluateMean(playerMean, meanOfMean, varOfMean, nSD)
    ```
    Description:
    Inputs:
    Outputs:
    ```
    if playerMean > meanOfMean + nSD*sqrt(varOfMean) # High mean performance
        meanStat = 2
    else if playerMean < meanOfMean - nSD*sqrt(varOfMean) # Low mean performance
        meanStat = 0
    else # Average mean performance
        meanStat = 1
    end
    return meanStat
end

function evaluateVariance(playerVar, meanOfVar, varOfVar, nSD)
    ```
    Description:
    Inputs:
    Outputs:
    ```
    if playerVar > meanOfVar + nSD*sqrt(varOfVar) # High variance in performance
        varStat = 2
    else if playerVar < meanOfVar - nSD*sqrt(varOfVar) # Low variance in performance
        varStat = 0
    else # Average variance in performance
        varStat = 1
    end
    return varStat
end

function compareLineupToLeague(lineup, data, nSD=1)
    # Define "High mean" as > 1 SD above league mean for that position (using varianceOfMean to define SD)
    # Define "High variance" as > 1 SD above league mean variance for that position (using varianceOfVariance to define SD)
    # Likewise, for "Low" and "Average"
    
    ```
    Description:
    Inputs:
    Outputs:
    ```

    leagueData = getLeaguePerformanceData(data) # Dictionary with ("Position", (meanOfMean, meanOfVar, varOfMean, varOfVar))
    lineupData = getLineupPerformanceData(lineup, data) # Dictionary with ("Position", (mean, var))

    relativeStats = Dict{String, Tuple}()

    for (pos, stats) in lineupData
        # Load data
        playerMean, playerVar = stats
        meanOfMean, meanOfVar, varOfMean, varOfVar = leagueData[pos]
        # Getting a value for the relative mean and variance of a player = (0,1,2) for (low, average, high)
        meanStat = evaluateMean(playerMean, meanOfMean, varOfMean, nSD)
        varState = evaluateVariance(playerVar, meanOfVar, varOfVar, nSD)

        
        relativeStats[pos] = (meanStat, varStat)
    end

end


# Define a performance function
# Currently very simple, can make this more complex
function performanceFxn(mean, variance)
    ```
    Description: Define a function that evaluates a player's  mean and variance and outputs a single value describing this
                 Needs to define what is "high, medium, or low" for mean and variance
                 Note: "high" for QB might be different than "high" for RB
                 0: "low"
                 1: "average"
                 2: "high"
    Inputs: Mean for a specific player (scalar)
            Variance for a specific player (scalar)
            
    Outputs: Scalar with range (00,22)
    ```
    

    performance_param = 
    return performance_param
end

# Need to store currently selected players' names
function namesToLineup(QB_name, RB_name, WR_name)
    ```
    Description: Convert three player names into a dictionary mapping their positions to their names
    Inputs: One player name for each position (string)
    Outputs: A dictionary storing the position and the associated name of the player -- Example: "QB" => "Tom Brady"
    ```
    lineup = Dict([("QB", QB_name), ("RB", RB_name), ("WR", WR_name)])
    return lineup
end

# Need to read from week's data to get player rankings
# Get player names, find names in the table, get stats, load to a dictionary
# current Week is going to be based on how the column in the dataframe is 
function getPlayerRankings(playerNames, data, currentWeek)
    ```
    Description:
    Inputs:
    Outputs:
    ```
    data
end




function makeRandomLineup(QBs, RBs, WRs, )
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

