% ------------------------------------------------------------------------
% Testing model-fits code for Experimnet 4 of the manuscript 
% "The nature of the perceptual representation for decision making".
%
% The code compares model-fits and uses the best-fitting parameters for 
% model testing. The testing results will be saved in 
% '.../Experiment 4/data/fitting_results'.
% To avoid overwriting the line 122 is currently commented. 
% To run this code, locate current directory to where the code is saved.
%
% Written by Jiwon Yeon. Last edited, Sep.14.2019.
% ------------------------------------------------------------------------

clear, clc
global data_sub

% Select model to test
modelToFit = 'summary';  %'population', 'summary'

% directories
dataPath = [fileparts(fileparts(fileparts(pwd))) '/data'];
savingPath = [dataPath '/fitting results/'];
fittingLists = dir([savingPath '/fitting_*.mat']);

% Lapse rate
lapse = 0;    % lapse rate

% Parameters needed for model fitting
N = 100000; % number of testing (100000)
nStim = 3;  % number of direction

% Load the data
load([dataPath '/subject_responses/dataForModeling']);

% compare AICs and select param_set
for nfit = 1:length(fittingLists)
    load([savingPath fittingLists(nfit).name]);    
    for subject = 1:length(modelFit)
        AIC(subject,nfit) = modelFit{subject}.AIC;
    end
end
[~, idx] = max(mean(AIC));
load([savingPath fittingLists(idx).name]);


%% testing
nSub = length(modelFit);
for subject = 1:nSub    
    data_sub.c3 = data.c3.correct(:,subject);
    data_sub.c2 = data.c2.correct(:,subject);
    paramSet = params(1,subject);
    logL = 0;
    
    % Define parameters
    mu  = [0, paramSet paramSet]'; % mu is 0 when it is the main direction
    
    % Generate simulated signal    
    signal = zeros(nStim,N);
    signal(:,1:floor(N*lapse)) = normrnd(0, 1, nStim, floor(N*lapse));
    signal(:,floor(N*lapse)+1:N) = normrnd(repmat(mu,1,N-floor(N*lapse)), ...
        ones(nStim,N-floor(N*lapse)), nStim, N-floor(N*lapse));

    %% Compute accuracy for 3-choice condition
    [~, responses] = max(signal);
    numCorrect = sum(responses==1);
    numWrong = N-numCorrect;
    p = numCorrect / N;
    p_c3(subject) = p;
    observed_p_c3(subject) = sum(data_sub.c3)/length(data_sub.c3);
    
    %% Compute accuracy for 2-choice condition
    % Suppose only available choices are direction1 and direction2
    if strcmp(modelToFit, 'population')
        [~, responses] = max(signal(1:2,:));
        numCorrect = sum(responses==1);
        
    elseif strcmp(modelToFit, 'summary')
        % If direction1 has highest activity in a trial, it is a correct trial
        [~, responses] = max(signal);
        numCorrect = sum(responses==1);  
        numWrong = sum(responses==2);
        
        % If either direction2 or direction3 is the highest, then randomly
        % choose a response
        numCorrect = numCorrect + (N-numWrong-numCorrect)/2;
    end
    p = numCorrect / N;
    p_c2(subject) = p;
    observed_p_c2(subject) = sum(data_sub.c2)/length(data_sub.c2);
    
    %% logL
    respProbability = [p_c2(subject) 1-p_c2(subject)];
    respObserved = [sum(data_sub.c2==1) sum(data_sub.c2==0)];
    for accuracy = 1:2
         if respProbability(accuracy)==0; respProbability(accuracy)=10^-10; end %replace probability of 0, in order to avoid -inf when taking log
         logL = logL - log(respProbability(accuracy)) * respObserved(accuracy);
    end
    
    logL = -logL;
    k    = length(paramSet);
    n    = length(data_sub.c3) + length(data_sub.c2);
    
    resfit{subject}.modelName = modelToFit;
    resfit{subject}.logL = logL;
    resfit{subject}.k    = k;
    resfit{subject}.n    = n;
    resfit{subject}.AIC  = -2*logL + 2*k;
    resfit{subject}.AICc = -2*logL + (2*k*n)/(n-k-1);
    resfit{subject}.BIC  = -2*logL + k*log(n);
    
end


%% Display results
observed = round([mean(observed_p_c3) mean(observed_p_c2)],3);
model = round([mean(p_c3); mean(p_c2)],3);
display_accuracy = table(observed', model, 'VariableNames', {'Oberved', strcat(modelToFit(1:3), '_Model')},...
    'RowName', {'3-choices', '2-choices'})


%% Save data
% save([savingPath '/' modelToFit], 'resfit', 'p_c3', 'p_c2')
