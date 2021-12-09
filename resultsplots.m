%% AA 228 Final Project Results Plots
%% resultsplots.m
%% 1 December 2021

close all
clear
clc


%% Read in raw data
dataR = {};
dataQ = {};
numyears = 2018-1999+1;
for i = 1:numyears
    year = 1998 + i;
    filenameR = ['EvaluatedData/' num2str(year) '_Random.csv'];
    filenameQ = ['EvaluatedData/' num2str(year) '_Qlearned.csv'];
    dataR{i} = csvread(filenameR,1,0); % row = iteration
    dataQ{i} = csvread(filenameQ,1,0); % col = week
end

avgR = []; % row = year
avgQ = []; % col = week

for i = 1:numyears
    matR = dataR{i}
    avgR(i,:) = mean(matR,1)
    matQ = dataQ{i}
    avgQ(i,:) = mean(matQ,1)
end


sumavgR = sum(avgR,2); 
sumavgQ = sum(avgQ,2);

%% Plots (the one actually used in the report)
figure;
plot(1999:2018,sumavgQ)
hold on
plot(1999:2018,sumavgR)

figure;
plot(2002:2012,sumavgQ(4:14))
hold on
plot(2002:2012,sumavgR(4:14))


figure;
bar(2002:2012,[sumavgQ(4:14),sumavgR(4:14)],'grouped')
title('Average Total Rewards vs Season')
legend('Learned Policy','Random Policy')
xlabel('NFL Season')
ylabel('Average Total Reward at End of Season')
%% Secondary metrics

for i = 4:14
    if i == 4
        cumsumavg(i-3) = sumavgQ(i) - sumavgR(i);
    else
        cumsumavg(i-3) = sumavgQ(i) - sumavgR(i) + cumsumavg(i-4);
    end
end

diffsumavg = sumavgQ(4:14) - sumavgR(4:14);
mean(diffsumavg)

sumsumavg = sumavgQ(4:14) + sumavgR(4:14);

percentdiff = diffsumavg./abs(sumsumavg);
percentchange = diffsumavg./sumavgR(4:14);
%
%% More Plots
figure;
plot(2002:2012,diffsumavg)
figure;
bar(2002:2012,diffsumavg)
title('Average Net Reward vs Season')
xlabel('Season')
ylabel('Average Net Reward')

figure;
plot(2002:2012,cumsumavg)
figure;
bar(2002:2012,cumsumavg)
title('Cumulative Average Net Reward vs Season')
xlabel('Season')
ylabel('Cumulative Average Net Reward')
%% Even more plots

figure;
bar(2002:2012,percentdiff)


figure;
bar(2002:2012,percentchange)

%figure;
%plot(avgQ(1,:))
%hold on
%plot(avgR(1,:))








