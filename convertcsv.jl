"""
The functions in this file were used to
generate the clean weekly and dirty weekly
datasets. Be careful and DON'T run them
if you don't want to risk overwriting
the current data.
"""

using DataFrames
using CSV
using Statistics
using Random

function rawInput(infile)
    rawWeekly = DataFrame(CSV.File(infile))
    return rawWeekly
end

function getPlayerNames(rawData)
    players = rawData.Player
    return players
end

function positionLists(data)
    QBlist = DataFrame()
    RBlist = DataFrame()
    WRlist = DataFrame()

    for i = 1:size(data,1)
        if ismissing(data[i,:].Pos)
        else
            if data[i,:].Pos == "QB"
                push!(QBlist,data[i,:])
            elseif data[i,:].Pos == "RB"
                push!(RBlist,data[i,:])
            elseif data[i,:].Pos == "WR"
                push!(WRlist,data[i,:])
            end
        end
    end

    return QBlist,RBlist,WRlist
end

function positionListsSorted(data)
    QBlist = DataFrame()
    RBlist = DataFrame()
    WRlist = DataFrame()

    for i = 1:size(data,1)
        if ismissing(data[i,:].Pos)
        else
            if data[i,:].Pos == "QB"
                push!(QBlist,data[i,:])
            elseif data[i,:].Pos == "RB"
                push!(RBlist,data[i,:])
            elseif data[i,:].Pos == "WR"
                push!(WRlist,data[i,:])
            end
        end
    end
    pointtype = 18
    sort!(QBlist,pointtype,rev=true)
    sort!(RBlist,pointtype,rev=true)
    sort!(WRlist,pointtype,rev=true)

    return QBlist,RBlist,WRlist
end

function positionListsSortShuffle(data)
    QBlist = DataFrame()
    RBlist = DataFrame()
    WRlist = DataFrame()

    for i = 1:size(data,1)
        if ismissing(data[i,:].Pos)
        else
            if data[i,:].Pos == "QB"
                push!(QBlist,data[i,:])
            elseif data[i,:].Pos == "RB"
                push!(RBlist,data[i,:])
            elseif data[i,:].Pos == "WR"
                push!(WRlist,data[i,:])
            end
        end
    end
    pointtype = 18
    sort!(QBlist,pointtype,rev=true)
    sort!(RBlist,pointtype,rev=true)
    sort!(WRlist,pointtype,rev=true)

    numSwaps = 4
    for i=1:numSwaps
        swap1 = rand(1:20)
        swap2 = rand(1:20)

        QBtemp = QBlist[swap1,:]
        QBlist[swap1,:] = QBlist[swap2,:]
        QBlist[swap2,:] = QBtemp

        RBtemp = RBlist[swap1,:]
        RBlist[swap1,:] = RBlist[swap2,:]
        RBlist[swap2,:] = RBtemp

        WRtemp = WRlist[swap1,:]
        WRlist[swap1,:] = WRlist[swap2,:]
        WRlist[swap2,:] = WRtemp

    end

    return QBlist,RBlist,WRlist
end

function switchNames(data,playernames)
    QBlist,RBlist,WRlist = positionLists(data)
    numPerPos = 20

    newData = DataFrame()

    for i=1:numPerPos
        push!(newData,QBlist[i,:])
    end
    for i=1:numPerPos
        push!(newData,RBlist[i,:])
    end
    for i=1:numPerPos
        push!(newData,WRlist[i,:])
    end

    for i=1:(numPerPos*3)
        newData[i,:].Player = playernames[i]
    end

    return newData
end

function switchNamesSorted(data,playernames)
    QBlist,RBlist,WRlist = positionListsSorted(data)
    numPerPos = 20

    newData = DataFrame()

    for i=1:numPerPos
        push!(newData,QBlist[i,:])
    end
    for i=1:numPerPos
        push!(newData,RBlist[i,:])
    end
    for i=1:numPerPos
        push!(newData,WRlist[i,:])
    end

    for i=1:(numPerPos*3)
        newData[i,:].Player = playernames[i]
    end

    return newData
end

function switchNamesSortShuffle(data,playernames)
    QBlist,RBlist,WRlist = positionListsSortShuffle(data)
    numPerPos = 20

    newData = DataFrame()

    for i=1:numPerPos
        push!(newData,QBlist[i,:])
    end
    for i=1:numPerPos
        push!(newData,RBlist[i,:])
    end
    for i=1:numPerPos
        push!(newData,WRlist[i,:])
    end

    for i=1:(numPerPos*3)
        newData[i,:].Player = playernames[i]
    end

    return newData
end

function cleanData(data1, data2)##
    if points[j] == 0 && i > 1
        points[j] = newData[j,week[max(i-1,1)]]
    end
end

function writenewCSV(newData,yearnum,weeknum)

    filepath = string("newweekly","//",yearnum,"//","week",weeknum,".csv")
    CSV.write(filepath, newData)
end

function writenewCSV2(newData,yearnum,weeknum)

    filepath = string("cleanweekly","//",yearnum,"//","week",weeknum,".csv")
    CSV.write(filepath, newData)
end

function writenewCSV3(newData,yearnum,weeknum)

    filepath = string("dirtyweekly","//",yearnum,"//","week",weeknum,".csv")
    CSV.write(filepath, newData)
end


function convert1()
    namespath = "2019//week1.csv"
    data2019 = rawInput(namespath)
    QB2019,RB2019,WR2019 = positionLists(data2019)
    allQBnames = getPlayerNames(QB2019)
    allRBnames = getPlayerNames(RB2019)
    allWRnames = getPlayerNames(WR2019)

    QBnames = allQBnames[1:20]
    RBnames = allRBnames[1:20]
    WRnames = allWRnames[1:20]

    playernames = vcat(QBnames,RBnames,WRnames)

    for year = 1999:2019
        pathstring = string(year)
        weeklist = readdir(pathstring)
        mkdir(string("newweekly//",pathstring))
        for i = 1:17
            fullpath = string(pathstring,"//",weeklist[i])
            data = rawInput(fullpath)
            newData = switchNames(data,playernames)
            writenewCSV(newData,year,i)
        end
    end
end

function convert2()
    for year = 1999:2019
        pathstring = string(year)
        weeklist = readdir(pathstring)
        mkdir(string("cleanweekly//",pathstring))
        for i = 1:17
            fullpath = string(pathstring,"//",weeklist[i])
            data = rawInput(fullpath)
            newData = cleanData(data,playernames)##
            writenewCSV(newData,year,i)##
        end
    end
end

function convert3()
    namespath = "2019/week1.csv"
    data2019 = rawInput(namespath)
    QB2019,RB2019,WR2019 = positionLists(data2019)
    allQBnames = getPlayerNames(QB2019)
    allRBnames = getPlayerNames(RB2019)
    allWRnames = getPlayerNames(WR2019)

    QBnames = allQBnames[1:20]
    RBnames = allRBnames[1:20]
    WRnames = allWRnames[1:20]

    playernames = vcat(QBnames,RBnames,WRnames)

    for year = 1999:2019
        pathstring = string(year)
        weeklist = readdir(pathstring)
        mkdir(string("cleanweekly//",pathstring))
        for i = 1:17
            fullpath = string(pathstring,"//",weeklist[i])
            data = rawInput(fullpath)
            newData = switchNamesSorted(data,playernames)
            writenewCSV2(newData,year,i)
        end
    end
end

function convert4()
    namespath = "2019/week1.csv"
    data2019 = rawInput(namespath)
    QB2019,RB2019,WR2019 = positionLists(data2019)
    allQBnames = getPlayerNames(QB2019)
    allRBnames = getPlayerNames(RB2019)
    allWRnames = getPlayerNames(WR2019)

    QBnames = allQBnames[1:20]
    RBnames = allRBnames[1:20]
    WRnames = allWRnames[1:20]

    playernames = vcat(QBnames,RBnames,WRnames)

    for year = 1999:2019
        pathstring = string(year)
        weeklist = readdir(pathstring)
        mkdir(string("dirtyweekly//",pathstring))
        for i = 1:17
            fullpath = string(pathstring,"//",weeklist[i])
            data = rawInput(fullpath)
            newData = switchNamesSortShuffle(data,playernames)
            writenewCSV3(newData,year,i)
        end
    end
end
