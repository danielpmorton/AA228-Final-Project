```
CODE OVERVIEW:

Main question: "How do we define the state of our Q-learning model"

Goals:
- Read from the tables generated in the CSV-reading/sorting code
- In the state, represent the current ranks of the players, as well as info on their 
  historical performance (mean and variance in points earned)
- Get the state into a scalar value so it is easy to index into the Q table

Final output:
state -- a numerical value defining the 9 information components we are looking at
      -> [QB_rank, RB_rank, WR_rank, QB_mean, RB_mean, WR_mean, QB_variance, RB_variance, WR_variance]
      (from the makeState function)

Some notes:
- As of 11/24: needs to be integrated better with Walter's code!
  -- note the structure of the data passed into getPlayerRankings() !!
```

using DataFrames
using CSV # Check if actually using this ones
using Statistics
using Random


# Define global variables
global positions= ["QB", "RB", "WR"]
global QBs = ["Jared Goff","Tom Brady","Dak Prescott","Deshaun Watson","Philip Rivers","Russell Wilson","Aaron Rodgers","Kirk Cousins","Matt Ryan","Derek Carr"]
global RBs = ["Ezekiel Elliott","Nick Chubb","Christian McCaffrey","Dalvin Cook","Chris Carson","Leonard Fournette","Mark Ingram","Todd Gurley","Alvin Kamara","Aaron Jones"]
global WRs = ["Michael Thomas","Keenan Allen","DeAndre Hopkins","Julio Jones","Allen Robinson","Tyler Lockett","Stefon Diggs","Chris Godwin","Mike Evans","Jarvis Landry"]


# Build the main function here, run it at the very end of the file
function main()
    lineup = namesToLineup("Tom Brady", "Christian McCaffrey", "Mike Evans") # OR: lineup = makeRandomLineup(QBs, RBs, WRs)
    performanceStats = compareLineupToLeague(lineup, data, nSD=1)
    ranks = getPlayerRankings(lineup, currentWeekData)
    state = makeState(ranks, performanceStats)
    print(state)
end

##### Begin sub functions #####

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
    Description: For each position, evaluate the mean and variance of the selected player
    Inputs: lineup: Dictionary containing the position and the associated name of the player
            data: The dataframe containing all of the data for the league (based on Walter's code) 
    Outputs: a dictionary containing the position and the stats for the chosen player as a tuple: "position" => (mean, variance)
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
    Description: Output 0,1,2 for low,average,high mean in performance wrt the league for that position
    Inputs: playerMean: Player's mean weekly score (scalar)
            meanOfMean: League average per-player score for that position (scalar)
            varOfMean: Variance in league average per-player score for that position(scalar)
            nSD: Number of standard deviations to define the cutoff points
    Outputs: 0, 1, or 2
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
    Description: Output 0,1,2 for low,average,high variance in performance wrt the league for that position
    Inputs: playerVar: Variance in player's weekly score (scalar)
            meanOfVar: League average variance for that position (scalar)
            varOfVar: Variance in league average variance for that position (scalar)
            nSD: Number of standard deviations to define the cutoff points
    Outputs: 0, 1, or 2
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
    ```
    Description: Compares each player's mean and variance to the league data and outputs a single value describing this
                 The "low", "medium", and "high" parameters are defined by standard deviations above/below the league mean
                 0: "low" -- player mean is >nSD SD below the league mean
                 1: "average" -- player mean is within nSD SD of the league mean
                 2: "high" -- player mean is >nSD SD above the league mean
                 This process is repeated for the variance as compared to the league
    Inputs: lineup: Dictionary containing the position and the associated name of the player
            data: The dataframe containing all of the data for the league (based on Walter's code) 
            nSD: The number of standard deviations to define the cutoffs in performance (default = 1)
    Outputs: Dictionary mapping "position" => (meanStat, varStat) which will be ((0, 1, or 2), (0, 1, or 2))
    ```
    
    leagueData = getLeaguePerformanceData(data) # Dictionary with ("Position", (meanOfMean, meanOfVar, varOfMean, varOfVar))
    lineupData = getLineupPerformanceData(lineup, data) # Dictionary with ("Position", (mean, var))

    performanceStats = Dict{String, Tuple}() # Initialize

    for (pos, stats) in lineupData
        # Load data
        playerMean, playerVar = stats
        meanOfMean, meanOfVar, varOfMean, varOfVar = leagueData[pos]
        # Getting a value for the relative mean and variance of a player = (0,1,2) for (low, average, high)
        meanStat = evaluateMean(playerMean, meanOfMean, varOfMean, nSD)
        varStat = evaluateVariance(playerVar, meanOfVar, varOfVar, nSD)
        # Save this to a dictionary: "position" => 2-entry tuple containing the 0,1,2 parameter for mean and variance
        performanceStats[pos] = (meanStat, varStat)
    end
    return performanceStats
end

function namesToLineup(QB_name, RB_name, WR_name)
    ```
    Description: Convert three player names into a dictionary mapping their positions to their names
    Inputs: One player name for each position (string)
    Outputs: A dictionary storing the position and the associated name of the player -- Example: "QB" => "Tom Brady"
    ```
    lineup = Dict([("QB", QB_name), ("RB", RB_name), ("WR", WR_name)])
    return lineup
end

# NOTE THE UNIQUE INPUT FORMAT IN THIS ONE
# Assumes we have a table like the others but with only one column of week data called "points" ****
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

function makeState(ranks, performanceStats)
    ```
    Description: Get an integer value based on the rank and the comparison 
    Inputs: ranks: Dictionary mapping "position" => rank in the league
            performanceStats: Dictionary mapping "position" => (meanStat, varStat) which will be ((0, 1, or 2), (0, 1, or 2))
    Outputs: state -- a numerical value defining the 9 information components we are looking at
                   -> [QB_rank, RB_rank, WR_rank, QB_mean, RB_mean, WR_mean, QB_variance, RB_variance, WR_variance]
    ```
    # Combine all of these numerical values as strings first -- outputs a 9-digit string
    # Note: for the rank, since we have 10 players, I subtracted one so this goes from 0 to 9 (instead of worrying about what happens when we have 2 digits in a rank)
    strung = string(ranks["QB"]-1) * string(ranks["RB"]-1) * string(ranks["WR"]-1) * 
        string(performanceStats["QB"][1]) * string(performanceStats["RB"][1]) * string(performanceStats["WR"][1]) * 
            string(performanceStats["QB"][2]) * string(performanceStats["RB"][2]) * string(performanceStats["WR"][2])
    
    # Un-string this value into a numerical value
    state = parse(Int64, strung)

    # Note: when parsing this, any of the leading terms being 0 will not be represented! But I think this will be ok

    return state
end

##### RUN THE MAIN FUNCTION #####
main() 