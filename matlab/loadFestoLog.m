function [DATE, X, Y, Z] = loadFestoLog(filename, H)

    %% Set up the Import Options and import the data
    opts = delimitedTextImportOptions("NumVariables", 4);

    % Specify range and delimiter
    opts.Delimiter = ";";

    % Specify column names and types
    opts.VariableNames = ["DATE", "X", "Y", "Z"];
    opts.VariableTypes = ["double", "double", "double", "double"];

    % Specify file level properties
    opts.ExtraColumnsRule = "ignore";
    opts.EmptyLineRule = "skip";
    opts.ImportErrorRule = "omitrow"; % WARNING: dangerous setting
    opts.ConsecutiveDelimitersRule = "split";
    opts.LeadingDelimitersRule = "ignore";

    % Import the data
    festoLog = readtable(filename, opts);

    festoLog.DATE = ...
        datetime(datevec(festoLog.DATE/1000/60/60/24), 'Format','yyyy/MM/dd-HH:mm:ss.SSS') + ...
        datenum('1-Jan-1970 00:00:00.000','dd-mmm-yyyy HH:MM:SS.FFF');

    if nargin > 1
        % Remove points beyond +-1mm from H
        mx = (abs(festoLog.Z - H) > 1000); 
        festoLog(mx,:) = [];
    end

    if nargout == 1
        % Correct unity of measure to get mm
        festoLog.X = festoLog.X/1000;
        festoLog.Y = festoLog.Y/1000;
        festoLog.Z = festoLog.Z/1000;

        DATE = festoLog;
    elseif nargout >= 2
        % Correct unity of measure to get mm
        X = festoLog.X/1000;
        Y = festoLog.Y/1000;
        Z = festoLog.Z/1000;

        DATE = festoLog.DATE;
    end

end 