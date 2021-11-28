using CSV
using DataFrames

"""
might be useful to clean data to eliminate bye/injury weeks
"""

function rawInput(infile)
    rawWeekly = DataFrame(CSV.File(infile))
    return rawWeekly
end

function getPlayerNames(rawData)
    players = rawData.Player
    return players
end

function getPositions(rawData)
    positions = rawData.Pos
    return positions
end

function createWeeklyData(year)

    # select statistic to track
    # 19 = HalfPPR, 18 = Standard, 17 = PPR
    pointType = 19

    pathstring = string("weekly//", year)
    weeklist = readdir(pathstring)
    numWeeks = 17

    week = ["week1","week2","week3","week4","week5","week6","week7",
            "week8","week9","week10","week11","week12","week13",
            "week14","week15","week16","week17"]
    newData = DataFrame();
    players = []
    positions = []

    for i = 1:numWeeks
        fullpath = string(pathstring,"//",weeklist[i])
        rawWeek = rawInput(fullpath)
        points = []
        if i == 1
            players = getPlayerNames(rawWeek)
            positions = getPositions(rawWeek)
            newData.player = players
            newData.position = positions
        end
        for j = 1:length(players)
            push!(points,0)
            for k = 1:size(rawWeek,1)
                if players[j] == rawWeek.Player[k]
                    points[j] = rawWeek[k,pointType]
                end
            end
        end
        newData[!,week[i]] = points
    end

    return newData
end

function createCleanWeeklyData(year)

    # select statistic to track
    # 19 = HalfPPR, 18 = Standard, 17 = PPR
    pointType = 19

    pathstring = string("weekly//", year)
    weeklist = readdir(pathstring)
    numWeeks = 17

    week = ["week1","week2","week3","week4","week5","week6","week7",
            "week8","week9","week10","week11","week12","week13",
            "week14","week15","week16","week17"]
    newData = DataFrame();
    players = []
    positions = []

    for i = 1:numWeeks
        fullpath = string(pathstring,"//",weeklist[i])
        rawWeek = rawInput(fullpath)
        points = []
        if i == 1
            players = getPlayerNames(rawWeek)
            positions = getPositions(rawWeek)
            newData.player = players
            newData.position = positions
        end
        for j = 1:length(players)
            push!(points,0)
            for k = 1:size(rawWeek,1)
                if players[j] == rawWeek.Player[k]
                    points[j] = rawWeek[k,pointType]
                end
            end
            if points[j] == 0 && i > 1
                points[j] = newData[j,week[max(i-1,1)]]
            end
        end
        newData[!,week[i]] = points
    end

    return newData
end

function selectGroup(subnames,weeklydata)
    subGroup = copy(weeklydata)

    deleteIndex = []

    for i = 1:size(subGroup,1)
        exclude = true
        for j = 1:length(subnames)
            if subGroup.player[i] == subnames[j]
                exclude = false
            end
        end
        if exclude == true
            push!(deleteIndex,i)
        end
    end

    delete!(subGroup,deleteIndex)
    return subGroup

end


years = readdir("weekly")

weekly2019 = createWeeklyData(years[end])
weekly2018 = createWeeklyData(years[end-1])

QBs = ["Jared Goff","Tom Brady","Dak Prescott","Deshaun Watson","Philip Rivers",
        "Russell Wilson","Aaron Rodgers","Kirk Cousins","Matt Ryan","Derek Carr"]

RBs = ["Ezekiel Elliott","Nick Chubb","Christian McCaffrey","Dalvin Cook","Chris Carson",
        "Leonard Fournette","Mark Ingram","Todd Gurley","Alvin Kamara","Aaron Jones"]

WRs = ["Michael Thomas","Keenan Allen","DeAndre Hopkins","Julio Jones","Allen Robinson",
        "Tyler Lockett","Stefon Diggs","Chris Godwin","Mike Evans","Jarvis Landry"]

subsetNames = vcat(QBs, RBs, WRs)

subset2019 = selectGroup(subsetNames,weekly2019)
subset2018 = selectGroup(subsetNames,weekly2018)


weekly2019clean = createCleanWeeklyData(years[end])
