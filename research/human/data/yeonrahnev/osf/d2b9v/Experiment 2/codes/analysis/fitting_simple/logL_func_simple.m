function logL = logL_func_simple(params)

% Get data from the workspace
global data_sub

% Figure out the mus
N   = 100000;
p   = params(1);
mu  = [0, params(2:end)]; %assume that mu for stimulus 1, when not dominant is 0
nStim = 6;

% Compute response probabilities and log likelihood
logL = 0;

% Simulate model activations
for stim=1:nStim
    for presented=0:1
        signal(stim,presented+1,1:floor(N*p)) = normrnd(0, 1, 1, floor(N*p)); %for these trials, all responses are random
        signal(stim,presented+1,floor(N*p)+1:N) = normrnd(mu(stim+nStim*presented), 1, 1, N-floor(N*p));
    end
end

for stimPresented=1:nStim
    % Determine the relevant signal in condition 1
    relevantSignal = squeeze(signal(:,1,:)); %all non-presented conditions
    relevantSignal(stimPresented,:) = signal(stimPresented,2,:); %use presented condition for stimPresented
    [~, response_cond1] = max(relevantSignal);
    
    % Loop over all responses (or pairs of stimuli)
    for stimRespOrPair=1:nStim
        
        % Compute respProbability separately for the 6-choice condition
        respProbability = sum(response_cond1==stimRespOrPair) / length(response_cond1);
        if respProbability==0; respProbability=10^-10; end %replace probability of 0, in order to avoid -inf when taking log
        logL = logL - log(respProbability) * data_sub.respPattern_cond1(stimPresented,stimRespOrPair);
        
    end
end
