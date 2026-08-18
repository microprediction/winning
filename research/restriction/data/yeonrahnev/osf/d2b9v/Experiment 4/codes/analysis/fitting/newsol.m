function params = newsol(params)

% Change only mu
% index = randi(length(params));
index = 1;

% Figure out by how much to alter that parameter
change = randn/10;
params(index) = params(index) + change;

%% Mute the line below since we don't fit lapse rate
% % If the parameter 1, make sure that it's not smaller than 0
% if index==1 && params(index) < 0
%     params(index) = params(index) - 2*change;
% end
