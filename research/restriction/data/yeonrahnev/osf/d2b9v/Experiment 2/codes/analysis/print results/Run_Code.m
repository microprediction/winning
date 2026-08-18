% ------------------------------------------------------------------------
% Data analysis code for the manuscript "The nature of the perceptual 
% representation for decision making".
%
% To run this code, locate your current directory to where the code is 
% saved. If you want to see the results of the main analysis (i.e., with 
% extended parameters) type 'extended' for the 'version' variable. If you 
% want to check the results of the simple version analysis (i.e., with
% fewer parameters) type 'simple' for the 'version' variable. 
%
% Written by Jiwon Yeon, last edited May 4th.2020.
% ------------------------------------------------------------------------
clear all, clc
version = 'extended';   % 'extended' or 'simple'

% load, organize, test, and print data
dataPath = fileparts(fileparts(fileparts(pwd)));
load([dataPath '/data/subject_responses/dataForModeling.mat'])
observed.alternative6 = data.acc(:,1);
observed.alternative2 = data.acc(:,2);

load([dataPath '/data/fitting results/' version '/population_' version '.mat'])
population.acc.alternative6 = accuracy_cond1;
population.acc.alternative2 = accuracy_cond2;
population.resfit = resfit;

load([dataPath '/data/fitting results/' version '/summary_' version '.mat'])
summary.acc.alternative6 = accuracy_cond1;
summary.acc.alternative2 = accuracy_cond2;
summary.resfit = resfit;

load([dataPath '/data/fitting results/' version '/twohighest_' version '.mat'])
twohighest.acc.alternative6 = accuracy_cond1;
twohighest.acc.alternative2 = accuracy_cond2;
twohighest.resfit = resfit;

load([dataPath '/data/fitting results/' version '/threehighest_' version '.mat'])
threehighest.acc.alternative6 = accuracy_cond1;
threehighest.acc.alternative2 = accuracy_cond2;
threehighest.resfit = resfit;

load([dataPath '/data/fitting results/attention_extended/attention_2.mat'])
att2.acc.alternative6 = accuracy_cond1;
att2.acc.alternative2 = accuracy_cond2;
att2.resfit = resfit;

load([dataPath '/data/fitting results/attention_extended/attention_3.mat'])
att3.acc.alternative6 = accuracy_cond1;
att3.acc.alternative2 = accuracy_cond2;
att3.resfit = resfit;


%% Accuracies
performance_6alt = round(mean([observed.alternative6, population.acc.alternative6, ...
    summary.acc.alternative6, twohighest.acc.alternative6, threehighest.acc.alternative6, ...
    att2.acc.alternative6, att3.acc.alternative6], 1),3);
performance_2alt = round(mean([observed.alternative2, population.acc.alternative2, ...
    summary.acc.alternative2, twohighest.acc.alternative2, threehighest.acc.alternative2, ...
    att2.acc.alternative2, att3.acc.alternative2], 1),3);
Performance = array2table([performance_6alt; performance_2alt], 'VariableNames', ...
    {'Observed', 'Population', 'Summary', 'TwoHighest', 'ThreeHighest', 'Attention2', 'Attention3'}, ...
    'RowNames', {'6_Alternative' '2_Alternative'})

n_performance_better_with_summary_compared_to_population_model = ...
    sum(abs(observed.alternative2-summary.acc.alternative2) ...
    <abs(observed.alternative2-population.acc.alternative2))

% Difference between Observed data and Attention models
Difference_between_Observed_vs_Attention_models = ...
    array2table(round([mean([att2.acc.alternative6-observed.alternative6, ...
    att3.acc.alternative6-observed.alternative6],1); ...
    mean([att2.acc.alternative2-observed.alternative2, ...
    att3.acc.alternative2- observed.alternative2],1)],3), 'VariableNames', {'Attention2', 'Attention3'},...
    'RowNames', {'6_Alternative', '2_Alternative'})

% t-test on performance
[h p ci stats] = ttest(population.acc.alternative2,observed.alternative2);
pop_ttest = [round(stats.tstat,3); p];
[h p ci stats] = ttest(summary.acc.alternative2,observed.alternative2);
sum_ttest = [round(stats.tstat,3); p];
[h p ci stats] = ttest(twohighest.acc.alternative2,observed.alternative2);
twoH_ttest = [round(stats.tstat,3); p];
[h p ci stats] = ttest(threehighest.acc.alternative2,observed.alternative2);
threeH_ttest = [round(stats.tstat,3); p];
[h p ci stats] = ttest(att2.acc.alternative2,observed.alternative2);
att2_ttest = [round(stats.tstat,3); p];
[h p ci stats] = ttest(att3.acc.alternative2,observed.alternative2);
att3_ttest = [round(stats.tstat,3); p];
ttest_2alternative_compare_to_observation = array2table([pop_ttest, sum_ttest, ...
    twoH_ttest, threeH_ttest, att2_ttest, att3_ttest],...
    'VariableNames', {'Population', 'Summary', 'TwoHighest', 'ThreeHighest', 'Attention2', 'Attention3'}, ...
    'RowNames', {'t_val', 'p_val'})

[h p ci stats] = ttest(twohighest.acc.alternative2,population.acc.alternative2);
pop_vs_twoH = [round(stats.tstat,3); p];
[h p ci stats] = ttest(twohighest.acc.alternative2,summary.acc.alternative2);
sum_vs_twoH = [round(stats.tstat,3); p];
[h p ci stats] = ttest(threehighest.acc.alternative2,population.acc.alternative2);
pop_vs_threeH = [round(stats.tstat,3); p];
[h p ci stats] = ttest(threehighest.acc.alternative2,summary.acc.alternative2);
sum_vs_threeH = [round(stats.tstat,3); p];
ttest_2alternative_compare_between_models = ...
    array2table([pop_vs_twoH sum_vs_twoH, pop_vs_threeH, sum_vs_threeH],...
    'VariableNames', {'Pop_vs_TwoHighest', 'Sum_vs_TwoHighest', 'Pop_vs_ThreeHihgest','Pop_vs_ThreeHighest'}, ...
    'RowNames', {'t_val', 'p_val'})




[h p ci stats] = ttest(att2.acc.alternative6,observed.alternative6);
att2_ttest = [round(stats.tstat,3); p];
[h p ci stats] = ttest(att3.acc.alternative6,observed.alternative6);
att3_ttest = [round(stats.tstat,3); p];
ttest_6alternative_compare_to_observation = array2table([att2_ttest, att3_ttest],...
    'VariableNames', {'Attention2', 'Attention3'}, ...
    'RowNames', {'t_val', 'p_val'})


%% AIC comparisons
for sub = 1:length(observed.alternative2)
    AIC.population(sub) = population.resfit{sub}.AIC;    
    AIC.summary(sub) = summary.resfit{sub}.AIC;
    AIC.twohighest(sub) = twohighest.resfit{sub}.AIC;
    AIC.threehighest(sub) = threehighest.resfit{sub}.AIC;
    AIC.att2(sub) = att2.resfit{sub}.AIC;
    AIC.att3(sub) = att3.resfit{sub}.AIC;
end

% number summary model preferred than population model
n_summary_model_preferred_via_AIC = sum(AIC.population > AIC.summary)
difference_greater_than_25_in_AIC_sum_vs_pop = sum(AIC.population-AIC.summary>25)

% Difference AIC_Average
AIC_difference_Average = array2table(round([mean(AIC.population-AIC.summary),...
    mean(AIC.twohighest-AIC.summary), mean(AIC.threehighest-AIC.summary)],3),...
    'VariableNames', {'Pop_vs_Sum', 'Sum_vs_TwoHighest', 'Sum_vs_ThreeHighest'})

% Compare AIC_Average
output = AICanalysis([mean(AIC.population) mean(AIC.summary)],'e');
output = [output; AICanalysis([mean(AIC.population) mean(AIC.threehighest)],'e')];
output = [output; AICanalysis([mean(AIC.population) mean(AIC.twohighest)],'e')];
Population = {output(1,1); output(2,1); output(3,1)};
output = AICanalysis([mean(AIC.threehighest) mean(AIC.summary)],'e');
ThreeHighest = {output(1,1); []; []};
output = AICanalysis([mean(AIC.twohighest) mean(AIC.summary)],'e');
TwoHighest = {output(1,1); []; []};
output = AICanalysis([mean(AIC.att2) mean(AIC.summary)],'e');
Attention2 = {output(1,1); []; []};
output = AICanalysis([mean(AIC.att3) mean(AIC.summary)],'e');
Attention3 = {output(1,1); []; []};
RowNames = {'Summary vs.'; 'TwoHighest vs'; 'ThreeHigest vs.'};
AIC_comparison_Average = table(Population, TwoHighest, ThreeHighest, ...
    Attention2, Attention3, 'RowNames', RowNames)


% Compare AIC_Total
diff_AIC_total = array2table(round([sum(AIC.population-AIC.summary), ...
    sum(AIC.twohighest-AIC.summary), sum(AIC.threehighest-AIC.summary)],3), ...
    'VariableNames', {'Pop_vs_Sum', 'Sum_vs_TwoHighest', 'Sum_vs_ThreeHighest'})

output = AICanalysis([sum(AIC.population) sum(AIC.summary)],'e');
output = [output; AICanalysis([sum(AIC.population) sum(AIC.threehighest)],'e')];
output = [output; AICanalysis([sum(AIC.population) sum(AIC.twohighest)],'e')];
Population = {output(1,1); output(2,1); output(3,1)};
output = AICanalysis([sum(AIC.threehighest) sum(AIC.summary)],'e');
ThreeHighest = {output(1,1); []; []};
output = AICanalysis([sum(AIC.twohighest) sum(AIC.summary)],'e');
TwoHighest = {output(1,1); []; []};
output = AICanalysis([sum(AIC.att2) sum(AIC.summary)],'e');
Attention2 = {output(1,1); []; []};
output = AICanalysis([sum(AIC.att3) sum(AIC.summary)],'e');
Attention3 = {output(1,1); []; []};
RowNames = {'Summary vs.'; 'TwoHighest vs'; 'ThreeHighest vs.'};
AIC_comparison_Total = table(Population, TwoHighest, ThreeHighest, ...
    Attention2, Attention3, 'RowNames', RowNames)



