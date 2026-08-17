function [params, logL, modelFit] = fitOneSub_attention(startingParamSet)

global data_sub

%% Make initial guess at parameter values
if ~exist('startingParamSet','var')
    mu = -1.5*ones(6,5);
    p = 0;
    startingParamSet = [p; mu(:)];
end


%% Perform maximum likelihood estimation and store parameters
op           = anneal();
op.Verbosity = 2;

op.Generator = @newsol;
[params, fval] = anneal(@logL_func_attention, startingParamSet, op);

logL = -fval;
k    = length(startingParamSet);
n    = sum(sum(data_sub.respPattern_cond1)) + sum(sum(sum(data_sub.respPattern_cond2)));

modelFit.logL = logL;
modelFit.k    = k;
modelFit.n    = n;
modelFit.AIC  = -2*logL + 2*k;
modelFit.AICc = -2*logL + (2*k*n)/(n-k-1);
modelFit.BIC  = -2*logL + k*log(n);

end