% helper function for getting a global variable
% useful for Matlab-compatible ContourPlots
% Return value:
% W_out = the value from the global variable W
function W_out = Get_Global_W
    global W
    W_out = W;