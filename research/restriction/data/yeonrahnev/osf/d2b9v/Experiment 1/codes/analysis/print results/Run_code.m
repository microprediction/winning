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
% Written by Jiwon Yeon, last edited May 4th, 2020.
% ------------------------------------------------------------------------
clear all, clc
version = 'extended';   % 'extended' or 'simple'

%load, organize, test, and print data
folderName = ['fitting results/' version];
dataPath = [fileparts(fileparts(fileparts(pwd))) '/data/'];
load([dataPath '/subject_responses/dataForModeling'])
observed.alternative4 = data.accuracy(:,1);
observed.alternative2 = data.accuracy(:,2);

load([dataPath folderName '/population_' version '.mat'])
population.acc.alternative4 = accuracy_cond1;
population.acc.alternative2 = accuracy_cond2;
population.resfit = resfit;

load([dataPath folderName '/summary_' version '.mat'])
summary.acc.alternative4 = accuracy_cond1;
summary.acc.alternative2 = accuracy_cond2;
summary.resfit = resfit;

load([dataPath folderName '/twohighest_' version '.mat'])
twohighest.acc.alternative4 = accuracy_cond1;
twohighest.acc.alternative2 = accuracy_cond2;
twohighest.resfit = resfit;

load([dataPath '/fitting results/attention_extended/attention_2.mat'])
attention2.acc.alternative4 = accuracy_cond1;
attention2.acc.alternative2 = accuracy_cond2;
attention2.resfit = resfit;

load([dataPath '/fitting results/attention_extended/attention_3.mat'])
attention3.acc.alternative4 = accuracy_cond1;
attention3.acc.alternative2 = accuracy_cond2;
attention3.resfit = resfit;

%% Print result for accuracy
accuracies = round([mean(observed.alternative4), mean(population.acc.alternative4),...
    mean(summary.acc.alternative4), mean(twohighest.acc.alternative4),...
    mean(attention2.acc.alternative4), mean(attention3.acc.alternative4)],3);
accuracies = [accuracies; [round(mean(observed.alternative2),3), ...
    round(mean(population.acc.alternative2),3), round(mean(summary.acc.alternative2),3), ...
    round(mean(twohighest.acc.alternative2),3), round(mean(attention2.acc.alternative2),3),...
    round(mean(attention3.acc.alternative2),3)]];
Accuracies = array2table(accuracies, 'VariableNames', {'Observed', ...
    'Population', 'Summary', 'Twohighest', 'Attention2', 'Attention3'}, ...
    'RowNames', {'4_Alternative', '2_Alternative'})

% t-test between models and observation
[h p ci stats] = ttest(population.acc.alternative2,observed.alternative2);
Population = {p; round(stats.tstat,3)};
[h p ci stats] = ttest(summary.acc.alternative2,observed.alternative2);
Summary = {p; round(stats.tstat,3)};
[h p ci stats] = ttest(twohighest.acc.alternative2,observed.alternative2);
TwoHighest = {p; round(stats.tstat,3)};
RowNames = {'p-val(vs. Observed)'; 't-val(vs. Observed)'};
ttest_2alternative = table(Population,Summary,TwoHighest,...
    'RowNames', RowNames)

[h p ci stats] = ttest(twohighest.acc.alternative2, population.acc.alternative2);
population_vs_TH = {p; round(stats.tstat,3)};
[h p ci stats] = ttest(twohighest.acc.alternative2, summary.acc.alternative2);
summary_vs_TH = {p; round(stats.tstat,3)};
ttest_2alternative_TwoHighest = array2table([population_vs_TH summary_vs_TH], ...
    'VariableNames', {'vs_population', 'vs_summary'}, 'RowNames', {'p_val', 't_val'})

% Comparison accuracies between population and summary model to observation
diff_pop_obs = population.acc.alternative2-observed.alternative2;
diff_sum_obs = summary.acc.alternative2-observed.alternative2;

model_vs_observation_accuracy = array2table(round([mean(diff_pop_obs) mean(abs(diff_pop_obs)) ...
    mean(diff_sum_obs) mean(abs(diff_sum_obs))],3), 'VariableNames', ...
    {'Pop_mean', 'Pop_absolute', 'Sum_mean', 'Sum_absolute'})

num_overPrediction = array2table([sum(abs(diff_pop_obs>0)) ...
    sum(abs(diff_pop_obs)>abs(diff_sum_obs))], 'VariableNames', ...
    {'Pop_vs_Obs', 'Sum_Better_Pop'})

[h p ci stats] = ttest(abs(diff_pop_obs),abs(diff_sum_obs));
ttest_population_vs_summary_models = array2table([round(stats.tstat,3), p], 'VariableNames', ...
    {'t_val', 'p_val'})

% Comparison accuracies between TwoHighest model to observation
diff_TH_obs = twohighest.acc.alternative2-observed.alternative2;
twoHighest_vs_observation_accuracy = ...
    array2table(round([mean(diff_TH_obs) mean(abs(diff_TH_obs))],4), 'VariableNames',...
    {'mean_difference', 'absoluate_difference'})

[h p ci stats] = ttest(twohighest.acc.alternative2, observed.alternative2);
TH_vs_Obs = [round(stats.tstat,3), p];
[h p ci stats] = ttest(abs(diff_TH_obs), abs(diff_sum_obs));
TH_vs_Sum = [round(stats.tstat,3) p];
ttest_twoHighest = array2table([TH_vs_Obs; TH_vs_Sum], 'VariableNames', ...
    {'t_val', 'p_val'}, 'RowNames', {'vs_Observation', 'vs_Summary'})

% Comparison accuracies between Attention models to observation - 4alternative
diff_att2_obs_4alt = attention2.acc.alternative4-observed.alternative4;
diff_att3_obs_4alt = attention3.acc.alternative4-observed.alternative4;
diff_att2_obs_2alt = attention2.acc.alternative2-observed.alternative2;
diff_att3_obs_2alt = attention3.acc.alternative2-observed.alternative2;

attention_model_accuracy_difference = array2table(round([mean(diff_att2_obs_4alt), ...
    mean(diff_att3_obs_4alt); mean(diff_att2_obs_2alt) mean(diff_att3_obs_2alt)],3) ,...
    'VariableNames', {'Attention2', 'Attention3'}, 'RowNames', ...
    {'4_Alternative', '2_Alternative'})


[h p ci stats] = ttest(attention2.acc.alternative4,observed.alternative4);
att2_obs_4alt = [round(stats.tstat,3), p];
[h p ci stats] = ttest(attention3.acc.alternative4,observed.alternative4);
att3_obs_4alt = [round(stats.tstat,3), p];
[h p ci stats] = ttest(attention2.acc.alternative2,observed.alternative2);
att2_obs_2alt = [round(stats.tstat,3), p];
[h p ci stats] = ttest(attention3.acc.alternative2,observed.alternative2);
att3_obs_2alt = [round(stats.tstat,3), p];
ttest_attention_to_observation = array2table([att2_obs_4alt, att3_obs_4alt; ...
    att2_obs_2alt, att3_obs_2alt], 'VariableNames', {'Attention2_tval', 'Attention2_pval', ...
    'Attention3_tval', 'Attention3_pval'}, 'RowNames', {'4_Alternative' '2_Alternative'})

%% AIC comparisons
for sub = 1:length(observed.alternative2)
    AIC.population(sub) = population.resfit{sub}.AIC;    
    AIC.summary(sub) = summary.resfit{sub}.AIC;
    AIC.twohighest(sub) = twohighest.resfit{sub}.AIC;
    AIC.att2(sub) = attention2.resfit{sub}.AIC;
    AIC.att3(sub) = attention3.resfit{sub}.AIC;
end

% Results
AIC_sum_over_pop = mean(AIC.population-AIC.summary);
AIC_sum_over_TH = mean(AIC.twohighest-AIC.summary);
AIC_average_difference_to_Summary_model = ...
    array2table([AIC_sum_over_pop AIC_sum_over_TH], 'VariableNames', ...
    {'Population', 'TwoHighest'})

AIC_total_sum_over_pop = sum(AIC.population-AIC.summary);
AIC_total_sum_over_TH = sum(AIC.twohighest-AIC.summary);
AIC_total_difference_to_Summary_model = ...
    array2table([AIC_total_sum_over_pop AIC_total_sum_over_TH], 'VariableNames', ...
    {'Population', 'TwoHighest'})

output = AICanalysis([mean(AIC.population) mean(AIC.summary)],'e');
output = [output; AICanalysis([mean(AIC.population) mean(AIC.twohighest)],'e')];
Population = {output(1,1); output(2,1)};
output = AICanalysis([mean(AIC.twohighest) mean(AIC.summary)],'e');
TwoHighest = {output(1,1); []};
output = AICanalysis([mean(AIC.att2) mean(AIC.summary)],'e');
Attention2 = {output(1,1); []};
output = AICanalysis([mean(AIC.att3) mean(AIC.summary)],'e');
Attention3 = {output(1,1); []};
RowNames = {'Summary vs.'; 'TwoHighest vs.'};
AIC_comparison_Average = table(Population, TwoHighest, Attention2, Attention3, ...
    'RowNames', RowNames)

output = AICanalysis([sum(AIC.population) sum(AIC.summary)],'e');
output = [output; AICanalysis([sum(AIC.population) sum(AIC.twohighest)],'e')];
Population = {output(1,1); output(2,1)};
output = AICanalysis([sum(AIC.twohighest) sum(AIC.summary)],'e');
TwoHighest = {output(1,1); []};
RowNames = {'Summary vs.'; 'TwoHighest vs.'};
AIC_comparison_Total = table(Population, TwoHighest, 'RowNames', RowNames)



