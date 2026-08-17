function params = newsol(params)

% Choose which parameter to alter
index = randi(length(params));

% Figure out by how much to alter that parameter
change = randn/10;
params(index) = params(index) + change;

% If the parameter 1, make sure that it's not smaller than 0
if index==1 && params(index) < 0
    params(index) = params(index) - 2*change;
end
