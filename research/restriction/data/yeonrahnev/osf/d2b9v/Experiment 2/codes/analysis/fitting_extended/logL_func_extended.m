function logL = logL_func_extended(params)

% Get data from the workspace
global data_sub

% Figure out the mus
N   = 100000; nStim = 6;
p   = params(1);
idx = ones(nStim,nStim) - eye(nStim,nStim);
idx = ind2sub(size(idx), find(idx==1));
mu = zeros(nStim,nStim);
mu(idx) = params(2:end);

% Compute response probabilities and log likelihood
logL = 0;

% Simulate model activations
signal = zeros(nStim,nStim,N);
for stimPresented=1:nStim
    mu_relevant = mu(:,stimPresented);
    signal(stimPresented,:,1:floor(N*p)) = normrnd(0, 1, nStim, floor(N*p)); %for these trials, all responses are random
    signal(stimPresented,:,floor(N*p)+1:N) = normrnd(repmat(mu_relevant,1,N-floor(N*p)), ...
        ones(nStim,N-floor(N*p)), nStim, N-floor(N*p));
end

for stimPresented=1:nStim
    % Determine the relevant signal in condition 1
    relevantSignal = signal(stimPresented,:,:);
    [~, response_cond1] = max(relevantSignal);
    
    % Loop over all responses (or pairs of stimuli)
    for stimRespOrPair=1:nStim
        
        % Compute respProbability separately for the 6-choice condition
        respProbability = sum(response_cond1==stimRespOrPair) / length(response_cond1);
        if respProbability==0; respProbability=10^-10; end %replace probability of 0, in order to avoid -inf when taking log
        logL = logL - log(respProbability) * data_sub.respPattern_cond1(stimPresented,stimRespOrPair);
        
    end
end
