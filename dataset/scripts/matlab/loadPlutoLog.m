function [DATE, FREQUENCY, I, Q] = loadPlutoLog(filename, dataLines)

    % If dataLines is not specified, define defaults
    if nargin < 2
        dataLines = [2, Inf];
    end

    %% Set up the Import Options and import the data
    opts = delimitedTextImportOptions("NumVariables", 9);

    % Specify range and delimiter
    opts.DataLines = dataLines;
    opts.Delimiter = " ";

    % Specify column names and types
    opts.VariableNames = ["DATE", "FREQUENCY_TX", "FREQUENCY_RX", "MOD_I", "PHASE_I", "MOD_Q", "PHASE_Q", "MOD", "PHASE"];
    opts.VariableTypes = ["datetime", "double", "double", "double", "double", "double", "double", "double", "double"];

    % Specify file level properties
    opts.ExtraColumnsRule = "ignore";
    opts.EmptyLineRule = "skip";
    opts.ConsecutiveDelimitersRule = "split";
    opts.LeadingDelimitersRule = "ignore";

    % Specify variable properties
    opts = setvaropts(opts, "DATE", "InputFormat", "yyyy/MM/dd-HH:mm:ss.SSSSSS");

    % Import the data
    plutoScan = readtable(filename, opts);

    % Send data to output variables
    if nargout == 1
        DATE = plutoScan;
    else
        if nargout >= 2
            DATE = plutoScan.DATE;
            FREQUENCY = plutoScan.FREQUENCY_TX;
        end
        if nargout >= 3
            I = plutoScan.MOD.*sin(plutoScan.PHASE);
        end
        if nargout >= 4
            Q = plutoScan.MOD.*cos(plutoScan.PHASE);
        end
    end

end