function logL = logL_func(params)

% Get data from the workspace
global data_sub

% Figure out the mus
N   = 100000;  % 100000
nStim = 3; % number of direction
lapse = params(end);
mu  = [0 params(1) params(1)]'; % mu is 0 when it is the main direction

% Compute response probabilities and log likelihood
logL = 0;

% Simulate model activations
signal = zeros(nStim,N);
signal(:,1:floor(N*lapse)) = normrnd(0, 1, nStim, floor(N*lapse));
signal(:,floor(N*lapse)+1:N) = normrnd(repmat(mu,1,N-floor(N*lapse)), ...
    ones(nStim,N-floor(N*lapse)), nStim, N-floor(N*lapse));

% Determine the relevant signal in condition 1 (the 4 sets of activations to choose from)
[~, response_c3] = max(signal);
    
% compute log likelihood
p = sum(response_c3==1) / length(response_c3);
respProbability = [p 1-p];
respObserved = [sum(data_sub.c3==1) sum(data_sub.c3==0)];
for accuracy = 1:2
    if respProbability(accuracy)==0; respProbability(accuracy)=10^-10; end %replace probability of 0, in order to avoid -inf when taking log
    logL = logL - log(respProbability(accuracy)) * respObserved(accuracy);
end
