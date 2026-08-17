function w = AICanalysis(AIC,type)
% w = AICanalysis(AIC,type)
%
% Given an array of AIC values, outputs AIC analysis of the specified type.
%
% input tokens for "type":
%
% 'd' : del(i) = AIC - min(AIC)
% 'l' : L(model|data) = exp(-.5 * del(i))
% 'w' : akaike weight = L(model|data) scaled to sum to 1
% 'e' : evidence ratio = L(model(best)|data) / L(model(i)|data)
%
% Default type is 'w'

if ~exist('type','var') || isempty(type), type = 'w'; end

minAIC = min(AIC);

switch type
    case 'd'
        w = AIC - min(AIC);
    case 'l'
        w = exp(-.5 * (AIC - min(AIC)) );
    case 'w'
        w = exp(-.5 * (AIC - min(AIC)) ) / sum(exp( -.5 * (AIC-min(AIC)) ));
    case 'e'
        w = exp( .5 * (AIC - min(AIC)) );
end