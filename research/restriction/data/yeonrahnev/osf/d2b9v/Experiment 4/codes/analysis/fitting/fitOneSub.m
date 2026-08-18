function [params, logL, modelFit] = fitOneSub(startingParamSet)

global data_sub

%% Make initial guess at parameter values
% if ~exist('startingParamSet','var')
%     mu = -1.5;
%     startingParamSet = mu;
% end

%% Perform maximum likelihood estimation and store parameters
op           = anneal();
op.Verbosity = 2;

op.Generator = @newsol;
[params, fval] = anneal(@logL_func, startingParamSet, op);

logL = -fval;
% k    = length(startingParamSet);
k = length(startingParamSet)-1;
n    = length(data_sub.c3) + length(data_sub.c2);

modelFit.logL = logL;
modelFit.k    = k;
modelFit.n    = n;
modelFit.AIC  = -2*logL + 2*k;
modelFit.AICc = -2*logL + (2*k*n)/(n-k-1);
modelFit.BIC  = -2*logL + k*log(n);

end