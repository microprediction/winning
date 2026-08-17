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
% Written by Jiwon Yeon, last edited Oct.12.2019.
% ------------------------------------------------------------------------
clear all, clc
version = 'extended';   % 'extended' or 'simple'

% load, organize, test, and print data
dataPath = fileparts(fileparts(fileparts(pwd)));
load([dataPath '/data/subject_responses/dataForModeling.mat'])
observed.answer1 = data.acc(:,1);
observed.answer2 = data.acc(:,2);

load([dataPath '/data/fitting results/' version '/population_' version '.mat'])
population.acc.answer1 = accuracy_cond1;
population.acc.answer2 = accuracy_cond2;
population.resfit = resfit;

load([dataPath '/data/fitting results/' version '/summary+random_' version '.mat'])
summary_random.acc.answer1 = accuracy_cond1;
summary_random.acc.answer2 = accuracy_cond2;
summary_random.resfit = resfit;

load([dataPath '/data/fitting results/' version '/summary+strategic_' version '.mat'])
summary_strategic.acc.answer1 = accuracy_cond1;
summary_strategic.acc.answer2 = accuracy_cond2;
summary_strategic.resfit = resfit;

load([dataPath '/data/fitting results/attention_extended/attention_2.mat'])
att2.acc.answer1 = accuracy_cond1;
att2.acc.answer2 = accuracy_cond2;
att2.resfit = resfit;

load([dataPath '/data/fitting results/attention_extended/attention_3.mat'])
att3.acc.answer1 = accuracy_cond1;
att3.acc.answer2 = accuracy_cond2;
att3.resfit = resfit;

%% Accuracy
answer1 = round(mean([observed.answer1, population.acc.answer1, summary_random.acc.answer1,...
    summary_strategic.acc.answer1, att2.acc.answer1, att3.acc.answer1],1),3);
answer2 = round(mean([observed.answer2, population.acc.answer2, summary_random.acc.answer2,...
    summary_strategic.acc.answer2, att2.acc.answer2, att3.acc.answer2],1),3);
Performance = array2table([answer1; answer2], 'VariableNames', ...
    {'Observed', 'Population', 'Summary_random', 'Summary_strategic', 'Attention2', 'Attention3'},...
    'RowNames', {'First_answer', 'Second_answer'})

diff_pop_obs = population.acc.answer2 - observed.answer2;
diff_sum_random_obs = summary_random.acc.answer2 - observed.answer2;
diff_sum_strategic_obs = summary_strategic.acc.answer2 - observed.answer2;
n_overestimation_performance = array2table([sum(diff_pop_obs>0) sum(diff_sum_random_obs>0), ...
    sum(diff_sum_strategic_obs>0)], 'VariableNames', {'Population', 'Summary_random', 'Summary_strategic'})

difference_performance_Attention_models = array2table(round(...
   [[mean(att2.acc.answer1-observed.answer1) mean(att3.acc.answer1-observed.answer1)]; ...
   [mean(att2.acc.answer2-observed.answer2) mean(att3.acc.answer2-observed.answer2)]],3), ...
    'VariableNames', {'Attention2', 'Attention3'}, 'RowNames', {'First_answer', 'Second_answer'})

% ttest on second answer
[h p ci stats] = ttest(population.acc.answer2,observed.answer2);
Population = {round((stats.tstat),3); p};
[h p ci stats] = ttest(summary_random.acc.answer2,observed.answer2);
Summary_Random = {round((stats.tstat),3); p};
[h p ci stats] = ttest(summary_strategic.acc.answer2,observed.answer2);
Summary_Strategic = {round((stats.tstat),3); p};
RowNames = {'t-val(vs. Observed)'; 'p-val(vs. Observed)'};
ttest_compare_to_observation = table(Population,Summary_Random,Summary_Strategic,...
    'RowNames', RowNames)

% ttest on Attention models
[h p ci stats] = ttest(att2.acc.answer1,observed.answer1);
Attention2_1stAns = {round((stats.tstat),3); p};
[h p ci stats] = ttest(att2.acc.answer2,observed.answer2);
Attention2_2ndAns = {round((stats.tstat),3); p};
[h p ci stats] = ttest(att3.acc.answer1,observed.answer1);
Attention3_1stAns = {round((stats.tstat),3); p};
[h p ci stats] = ttest(att3.acc.answer2,observed.answer2);
Attention3_2ndAns = {round((stats.tstat),3); p};
RowNames = {'t-val(vs. Observed)'; 'p-val(vs. Observed)'};
ttest_compare_to_observation_attention_models = table(Attention2_1stAns, Attention2_2ndAns, ...
    Attention3_1stAns, Attention3_2ndAns, 'RowNames', RowNames)


%% AIC comparisons
for sub = 1:length(observed.answer2)
    AIC.population(sub) = population.resfit{sub}.AIC;    
    AIC.summary_random(sub) = summary_random.resfit{sub}.AIC;
    AIC.summary_strategic(sub) = summary_strategic.resfit{sub}.AIC;
    AIC.att2(sub) = att2.resfit{sub}.AIC;
    AIC.att3(sub) = att3.resfit{sub}.AIC;
end

% Compare AIC_Average
AIC_difference_Average = array2table(round([mean(AIC.population-AIC.summary_random),...
    mean(AIC.population-AIC.summary_strategic), mean(AIC.att2-AIC.summary_random), ...
    mean(AIC.att3-AIC.summary_strategic)],3),...
    'VariableNames', {'Pop_vs_Sum_random', 'Pop_vs_Sum_strategic', ...
    'Attention2_vs_Sum_random', 'Attention3_vs_Sum_random'})

output = AICanalysis([mean(AIC.population) mean(AIC.summary_random)],'e');
output = [output; AICanalysis([mean(AIC.population) mean(AIC.summary_strategic)],'e')];    
Population = {output(1,1); output(2,1)};
output = AICanalysis([mean(AIC.summary_random) mean(AIC.summary_strategic)],'e');
Summary_Random = {[]; output(1,1)};
output = AICanalysis([mean(AIC.summary_random) mean(AIC.att2)],'e');
Attention2 = {output(1,2);[]};
output = AICanalysis([mean(AIC.summary_random) mean(AIC.att3)],'e');
Attention3 = {output(1,2);[]};
RowNames = {'Summary_Random vs.'; 'Summary_Strategic vs.'};
AIC_comparison_Average = table(Population, Summary_Random, Attention2, ...
    Attention3, 'RowNames', RowNames)


% Compare AIC_Average
AIC_difference_Total = array2table(round([sum(AIC.population-AIC.summary_random),...
    sum(AIC.population-AIC.summary_strategic), sum(AIC.att2-AIC.summary_random), ...
    sum(AIC.att3-AIC.summary_strategic)],3),...
    'VariableNames', {'Pop_vs_Sum_random', 'Pop_vs_Sum_strategic', ...
    'Attention2_vs_Sum_random', 'Attention3_vs_Sum_random'})

output = AICanalysis([sum(AIC.population) sum(AIC.summary_random)],'e');
output = [output; AICanalysis([sum(AIC.population) sum(AIC.summary_strategic)],'e')];    
Population = {output(1,1); output(2,1)};
output = AICanalysis([sum(AIC.summary_random) sum(AIC.summary_strategic)],'e');
Summary_Random = {[]; output(1,1)};
RowNames = {'Summary_Random vs.'; 'Summary_Strategic vs.'};
AIC_comparison_Total = table(Population, Summary_Random, ...
    'RowNames', RowNames)

