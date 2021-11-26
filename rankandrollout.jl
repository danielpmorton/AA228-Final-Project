using DataFrames
using CSV
using Statistics
using Random

"""
 This function can be used for exploration.
 It currently generates a random lineup and calculates its performance for
 a random week in a given year.
 It still needs the actions and rewards to be defined.
 It also might need to be checked for compatibility with Daniel's functions.
"""
function simulateRandomLineup(year)

    # input data
    weeklydata = createWeeklyData(year)

    QBlist = []
    RBlist = []
    WRlist = []

    # get list of player names for each position to use for sorting
    for i = 1:size(weeklydata)
        if weeklydata[i].position == "QB"
            push!(QBlist,weeklydata[i].player)
        elseif weeklydata[i].position == "RB"
            push!(RBlist,weeklydata[i].player)
        elseif weeklydata[i].position == "WR"
            push!(WRlist,weeklydata[i].player)
        end
    end

    # pick random subset of 10 players at each position
    shuffleQBlist = randcycle(length(QBlist))
    shuffleRBlist = randcycle(length(RBlist))
    shuffleWRlist = randcycle(length(WRlist))

    QBsubset = QBlist[shuffleQBlist[1:10]]
    RBsubset = RBlist[shuffleRBlist[1:10]]
    WRsubset = WRlist[shuffleWRlist[1:10]]

    # random lineup
    QBstart = rand(1:length(QBsubset))
    RBstart = rand(1:length(RBsubset))
    WRstart = rand(1:length(WRsubset))

    QBdata = []
    RBdata = []
    WRdata = []

    # generate data subsets
    for i = 1:10
            QBidx = findall(weeklydata.player .== QBsubset[i])
            RBidx = findall(weeklydata.player .== RBsubset[i])
            WRidx = findall(weeklydata.player .== WRsubset[i])

            push!(QBdata,weeklydata[QBidx])
            push!(RBdata,weeklydata[RBidx])
            push!(WRdata,weeklydata[WRidx])
    end

    # calculate rankings
    startweek = rand(2:17)
    sort!(QBdata,[2+startweek],rev = true)
    sort!(RBdata,[2+startweek],rev = true)
    sort!(WRdata,[2+startweek],rev = true)

    # set lineup
    lineup_start = [QBstart RBstart WRstart]
    names_start = [QBdata[QBstart].player RBdata[RBstart].player WRdata[WRstart].player]

     # calculate first week fantasy point totals
    points_start = QBdata[QBstart,2+startweek] + RBdata[RBstart,2+startweek] + WRdata[WRstart,2+startweek]

    action = rand(1:7)

# need to add actual actions here
    if action == 1 # swap QB up
            lineup_temp = lineup_start
            names_new = [QBdata[lineup_temp[1]].player names_start[2] names_start[3]]

    elseif action == 2 # swap QB down
            lineup_temp = lineup_start
            names_new = [QBdata[lineup_temp[1]].player names_start[2] names_start[3]]

    elseif action == 3 # swap RB up
            lineup_temp = lineup_start
            names_new = [names_start[1] RBdata[lineup_temp[2]].player names_start[3]]

    elseif action == 4 # swap RB down
            lineup_temp = lineup_start
            names_new = [names_start[1] RBdata[lineup_temp[2]].player names_start[3]]

    elseif action == 5 # swap WR up
            lineup_temp = lineup_start
            names_new = [names_start[1] names_start[2] WRdata[lineup_temp[3]].player]

    elseif action == 6 # swap WR down
            lineup_temp = lineup_start
            names_new = [names_start[1] names_start[2] WRdata[lineup_temp[3]].player]

    elseif action == 7 # do nothing
            lineup_temp = lineup_start
            names_new = names_start
    end

    # create new rankings
    sort!(QBdata,[2+startweek+1],rev = true)
    sort!(RBdata,[2+startweek+1],rev = true)
    sort!(WRdata,[2+startweek+1],rev = true)

    # create new lineup state
    QBnew = findall(QBdata.player .== names_new[1])
    RBnew = findall(RBdata.player .== names_new[2])
    WRnew = findall(WRdata.player .== names_new[3])

    lineup_new = [QBnew RBnew WRnew]

    # calculate new fantasy point totals
    points_new = QBdata[QBnew,2+startweek+1] + RBdata[RBnew,2+startweek+1] + WRdata[WRnew,2+startweek+1]


    # need to add reward function here
    # need to decide which vars to return

end

# Generates dataframe with all states
function allStates()
        states = DataFrame(QB = [], RB = [], WR = [])
        for i = 1:10
                for j = 1:10
                        for k = 1:10
                                state = [i; j; k]
                                push!(states,state)
                        end
                end
        end
        return states
end

# Generates dataframe containing random Policy (7 actions)
function randomPolicy()
        states = allStates()
        actions = []
        for i = 1:size(states,1)
                act = rand(1:7)
                push!(actions,act)
        end
        policy = states
        policy.action = actions
        return policy
end

"""
step through each week of season and
calculate total fantasy points given a policy
*still in progress, need to incorporate Daniel's functions
"""
function simulateSeason(policy,year)
        #randomlineup
        weeklydata = createWeeklyData(year)
        points = []

        for i=1:17
                score = QBdata[QBnew,2+i] + RBdata[RBnew,2+i] + WRdata[WRnew,2+i]
                points = push!(points,score)
        end

        return points
end
