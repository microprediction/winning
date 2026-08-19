% ------------------------------------------------------------------------
% Data analysis code for Experiment 4 of the manuscript
% "The nature of the perceptual representation for decision making".
%
% To run this code, locate your current directory to where the code is 
% saved. 
%
% Written by Jiwon Yeon, last edited Oct.12.2019.
% ------------------------------------------------------------------------

clear all, clc
dataPath = [fileparts(fileparts(fileparts(pwd))) '/data'];

%% load, organize, test, and print data
folderName = [dataPath '/subject_responses/' ];
load([folderName 'dataForModeling.mat'])
observed.choice3 = mean(data.c3.correct,1);
observed.choice2 = mean(data.c2.correct,1);

folderName = [dataPath '/fitting results/'];
load([folderName 'population.mat'])
population.acc.choice3 = p_c3;
population.acc.choice2 = p_c2;
population.resfit = resfit;

load([folderName 'summary.mat'])
summary.acc.choice3 = p_c3;
summary.acc.choice2 = p_c2;
summary.resfit = resfit;

%% Accuracy
% 3choices
cols = {'Observed', 'Population', 'Summary'};
t_input = [mean(observed.choice3) mean(population.acc.choice3) mean(summary.acc.choice3)];
Prediction_3choice_accuracy = array2table(t_input,'variablenames',cols)

% 2choice
Observed = {round(mean(observed.choice2),3); []; []};
[h p ci stats] = ttest(observed.choice2, population.acc.choice2);
Population = {round(mean(population.acc.choice2),3); p; round(abs(stats.tstat),3)};
[h p ci stats] = ttest(observed.choice2, summary.acc.choice2);
Summary = {round(mean(summary.acc.choice2),3); p; round(abs(stats.tstat),3)};

RowNames = {'Accuracy'; 'p-val(vs. Observed)'; 't-val(vs. Observed)'};
Prediction_2choice = table(Observed,Population,Summary, ...
    'RowNames', RowNames)

n_summary_perform_better_than_population = sum(abs(observed.choice2-summary.acc.choice2) ...
    < abs(observed.choice2-population.acc.choice2))

%% AIC
for sub = 1:length(observed.choice2)
    AIC.population(sub) = population.resfit{sub}.AIC;    
    AIC.summary(sub) = summary.resfit{sub}.AIC;
end

AIC_difference_Average = round(mean(AIC.population - AIC.summary),3)
AIC_difference_Total = round(sum(AIC.population - AIC.summary),3)

output = AICanalysis([mean(AIC.population), mean(AIC.summary)],'e');
mean_difference = output(1);
output = AICanalysis([sum(AIC.population), sum(AIC.summary)],'e');
total_difference = output(1);
AIC_comparisons = array2table([mean_difference;total_difference], 'variablenames',{'Population_vs_Summary'},...
    'rownames', {'Average', 'Total'})
