function logL = logL_func_attention(params)

% Get data from the workspace
global data_sub

% Figure out the mus
nItem = data_sub.nItem;
N   = 100000;
nStim = 6;
p   = params(1);
idx = ones(nStim,nStim)-eye(nStim,nStim);
idx = ind2sub(size(idx),find(idx==1));
mu = zeros(nStim,nStim);
mu(idx) = params(2:end);

% Compute response probabilities and log likelihood
logL = 0;

% Simulate model activations
signal = zeros(nStim, nStim, N);
for stimPresented=1:nStim
    mu_relevant = mu(:,stimPresented);
    signal(stimPresented,:,1:floor(N*p)) = normrnd(0, 1, nStim, floor(N*p)); %for these trials, all responses are random
    signal(stimPresented,:,floor(N*p)+1:N) = normrnd(repmat(mu_relevant,1,N-floor(N*p)), ...
        ones(nStim,N-floor(N*p)), nStim, N-floor(N*p));
end

% Consider all possible pairs
for stimPresented=1:nStim
    % Generate activation for each stimulus
    relevantSignal = squeeze(signal(stimPresented,:,:));
    
    % select attended item/stimulus
    [~, att_item] = sort(rand(N,nStim),2);
    att_item = att_item(:,1:nItem)';
    
    % select attended signals and decide response
    attendedSignal(1,:) = relevantSignal(sub2ind(size(relevantSignal),att_item(1,:),[1:N]));
    attendedSignal(2,:) = relevantSignal(sub2ind(size(relevantSignal),att_item(2,:),[1:N]));
    if nItem == 3
        attendedSignal(3,:) = relevantSignal(sub2ind(size(relevantSignal),att_item(3,:),[1:N]));
    end
    [~, response] = max(attendedSignal);
    response = att_item(sub2ind(size(att_item),response,[1:N]));
    
    % Loop over all responses 
    for stimResp=1:nStim
        respProbability = sum(response==stimResp) / length(response);
        if respProbability==0; respProbability=10^-10; end %replace probability of 0, in order to avoid -inf when taking log
        logL = logL - log(respProbability) * data_sub.respPattern_cond1(stimPresented,stimResp);
    end
end