using DataFrames
using CSV
using Statistics
using Random


function updateRanks(lineup,rankings,weeknum,data)



    return new_lineup, new_rankings
end

function simulateRandomLineup(year)

    weeklydata = createWeeklyData(year)

    QBlist = []
    RBlist = []
    WRlist = []

    for i = 1:size(weeklydata)
        if weeklydata[i].postion = "QB"
            push!(QBlist,weeklydata[i].player)
        elseif weeklydata[i].postion = "RB"
            push!(RBlist,weeklydata[i].player)
        elseif weeklydata[i].postion = "WR"
            push!(WRlist,weeklydata[i].player)
    end

    shuffleQBlist = randcycle(length(QBlist))
    shuffleRBlist = randcycle(length(RBlist))
    shuffleWRlist = randcycle(length(WRlist))

    QBsubset = QBlist[shuffleQBList[1:10]]
    RBsubset = RBlist[shuffleRBList[1:10]]
    WRsubset = WRlist[shuffleWRList[1:10]]

    QBstart = rand(1:length(QBsubset))
    RBstart = rand(1:length(RBsubset))
    WRstart = rand(1:length(WRsubset))

    QBdata = []
    RBdata = []
    WRdata = []

    for i = 1:10
            QBidx = findall(weeklydata.player .== QBsubset[i])
            RBidx = findall(weeklydata.player .== RBsubset[i])
            WRidx = findall(weeklydata.player .== WRsubset[i])

            push!(QBdata,weeklydata[QBidx])
            push!(RBdata,weeklydata[RBidx])
            push!(WRdata,weeklydata[WRidx])
    end

    startweek = rand(2:17)
    sort!(QBdata,[2+startweek],rev = true)
    sort!(RBdata,[2+startweek],rev = true)
    sort!(WRdata,[2+startweek],rev = true)

    lineup_start = [QBstart RBstart WRstart]
    names_start = [QBdata[QBstart].player RBdata[RBstart].player WRdata[WRstart].player]

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

    sort!(QBdata,[2+startweek+1],rev = true)
    sort!(RBdata,[2+startweek+1],rev = true)
    sort!(WRdata,[2+startweek+1],rev = true)

    QBnew = findall(QBdata.player .== names_new[1])
    RBnew = findall(RBdata.player .== names_new[2])
    WRnew = findall(WRdata.player .== names_new[3])

    lineup_new = [QBnew RBnew WRnew]

    points_new = QBdata[QBnew,2+startweek+1] + RBdata[RBnew,2+startweek+1] + WRdata[WRnew,2+startweek+1]


    # need to add reward function here
    # need to decide which vars to return

end