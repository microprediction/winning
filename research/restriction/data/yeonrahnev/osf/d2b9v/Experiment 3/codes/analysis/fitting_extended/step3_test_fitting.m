% ------------------------------------------------------------------------
% Testing model-fits code for Experimnet 3 of the manuscript 
% "The nature of the perceptual representation for decision making".
%
% The code compares model-fits and uses the best-fitting parameters for 
% model testing. The testing results will be saved in 
% '.../Experiment 3/data/fitting results/extended'.
% To avoid overwriting, the line 142 is currently commented. 
% To run this code, locate current directory to where the code is saved.
%
% Written by Jiwon Yeon. Last edited, Jan.23.2019.
% ------------------------------------------------------------------------

clear, clc

% Select model to test
modelToFit = 'population';  %'population', 'summary+random', 'summary+strategic'

% directories
dataPath = [fileparts(fileparts(fileparts(pwd))) '/data/subject_responses'];
savingPath = [fileparts(dataPath) '/fitting results/extended/'];
fitDataList = dir([savingPath '/fit*.mat']);

% Parameters needed for model fitting
N = 100000; % number of testing
nStim = 6;  % number of stimuli

% Load the data
load([dataPath '/dataForModeling']);

% compare AICs and select param_set
for nfit = 1:length(fitDataList)
    load([savingPath fitDataList(nfit).name]);    
    for subject = 1:length(modelFit)
        AIC(subject,nfit) = modelFit{subject}.AIC;
    end
end
[~, idx] = max(mean(AIC));
load([savingPath fitDataList(idx).name]);

%% testing
for subject = 1:length(modelFit)
    data_sub.respPattern_cond1 = squeeze(data.respPattern_cond1(subject,:,:));
    data_sub.respPattern_cond2 = squeeze(data.respPattern_cond2(subject,:,:,:));    
    paramSet = params(:,subject);
    logL = 0;
    
    % Define parameters
    p   = paramSet(1);
    idx = ones(nStim,nStim)-eye(nStim,nStim);
    idx = ind2sub(size(idx),find(idx==1));
    mu = zeros(nStim,nStim);
    mu(idx) = paramSet(2:end);

    % Simulate model activations
    for stimPresented=1:nStim
        mu_relevant = mu(:,stimPresented);
        signal(stimPresented,:,1:floor(N*p)) = normrnd(0, 1, nStim, floor(N*p)); %for these trials, all responses are random
        signal(stimPresented,:,floor(N*p)+1:N) = normrnd(repmat(mu_relevant,1,N-floor(N*p)), ...
            ones(nStim,N-floor(N*p)), nStim, N-floor(N*p));
    end
    
    % simulation
    for stimPresented=1:nStim
        relevantSignal = squeeze(signal(stimPresented,:,:));
        relevantPattern = squeeze(data.respPattern_cond2(subject,stimPresented,:,:));
        
        %%%% find first choice %%%%        
        [maxval, firstchoice] = max(relevantSignal);
        numCorrect1 = sum(firstchoice == stimPresented);
        numWrong = sum(firstchoice ~= stimPresented);        
        p_cond1(subject,stimPresented) = numCorrect1 / N;
        
        %%%%%% find second choice %%%%%%
        if strcmp(modelToFit, 'population')
            maxval = repmat(maxval,[6,1]);
            relevantSignal(relevantSignal == maxval) = NaN;
            [~, secondchoice] = max(relevantSignal);
            secondchoice(firstchoice == stimPresented) = [];
            numCorrect2 = sum(secondchoice == stimPresented);
            
            % create pattern response from the model
            choices = [firstchoice(firstchoice~=stimPresented)', secondchoice'];
            modelPattern = zeros(nStim,nStim);
            idx = sub2ind(size(modelPattern),choices(:,1),choices(:,2));            
            for location = 1:36
                modelPattern(location) = length(find(idx==location));
            end            
            modelPattern = modelPattern ./ repmat(sum(modelPattern,2), [1,nStim]);            
            modelPattern(stimPresented,:) = 0; % remove NaN
            
        elseif strcmp(modelToFit, 'summary+random')
            numCorrect2 = numWrong/(nStim-1);
            modelPattern = (ones(nStim,nStim)-eye(nStim,nStim))*numCorrect2;
            modelPattern = modelPattern ./ (repmat(sum(modelPattern,2),[1,nStim]));
            modelPattern(stimPresented,:) = 0;
            
        elseif strcmp(modelToFit, 'summary+strategic')
            freq_nontarget = 7/(49-7);
            freq_target = 14/(49-7);
            numCorrect2 = numWrong*freq_target;
            modelPattern = (ones(nStim,nStim)-eye(nStim,nStim))*freq_nontarget;
            modelPattern(:,stimPresented) = freq_target;
            modelPattern(stimPresented,:) = 0;
            
        end
        p_cond2(subject,stimPresented) = numCorrect2 / numWrong;
                    
        %%%%%%% logL %%%%%%
        modelPattern(modelPattern==0) = 10^-10;        
        gain = sum(sum(log(modelPattern) .* relevantPattern));
        logL = logL - gain;                
    end
    
    %%%%%% accuracy(unbalanced) %%%%%%
    accuracy_cond1(subject,1) = (p_cond1(subject,:)*sum(data_sub.respPattern_cond1,2))/sum(sum(data_sub.respPattern_cond1));
    accuracy_cond2(subject,1) = (p_cond2(subject,:)*sum(sum(data_sub.respPattern_cond2,3),2))/sum(sum(sum(data_sub.respPattern_cond2)));
   
    % organize data to save
    logL = -logL;
    k    = length(paramSet);
    n    = sum(sum(data_sub.respPattern_cond1)) + sum(sum(sum(data_sub.respPattern_cond2)));
    
    resfit{subject}.modelName = modelToFit;
    resfit{subject}.logL = logL;
    resfit{subject}.k    = k;
    resfit{subject}.n    = n;
    resfit{subject}.AIC  = -2*logL + 2*k;
    resfit{subject}.AICc = -2*logL + (2*k*n)/(n-k-1);
    resfit{subject}.BIC  = -2*logL + k*log(n);

end


%% Display results
observed = round([mean(data.acc(:,1)); mean(data.acc(:,2))],3);
model = round([mean(accuracy_cond1); mean(accuracy_cond2)],3);
display_accuracy = table(observed, model, 'RowNames', {'1st answer'; '2nd answer'})


%% Save data
% save([savingPath '/' modelToFit '_extended'], 'resfit', 'accuracy_cond1', 'accuracy_cond2')