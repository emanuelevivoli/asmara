function [P_F_UNIQUE, FFMOD, FFPHASE, P_X, P_Y, P_I, P_Q] = merge_acquisition(FFESTO,FPLUTO)

    % WARNING: constants in code
    NEIG = 'linear'; % ExtrapolationMethod can be: 'nearest', 'linear', or 'none'
    INTM = 'natural'; % Interpolation method can be: 'nearest', 'linear', or 'natural'.
    TOFF = 0; % Delay in ms applied to PLUTO scan data timings

    if nargin < 3
        P1 = loadFestoLog(FFESTO);
        [co,ce]=histcounts(P1.Z,1000);
        [~,cx] = max(co);
        H = ce(cx)*1000;
        clear co ce cx
    P1 = loadFestoLog(FFESTO, H);

    P2 = loadPlutoLog(FPLUTO);
    P2.DATE = P2.DATE + seconds(TOFF*1e-3);

    if any(abs(P2.MOD_I.*sin(P2.PHASE_I)) >= 2^11-1 | ...
            abs(P2.MOD_I.*cos(P2.PHASE_I)) >= 2^11-1)
        disp('WARNING: signal I is saturated.');
    end
    if any(abs(P2.MOD_Q.*sin(P2.PHASE_Q)) >= 2^11-1 | ...
            abs(P2.MOD_Q.*cos(P2.PHASE_Q)) >= 2^11-1)
        disp('WARNING: signal Q is saturated.');
    end

    % Index of PLUTO's data within FESTO's time window
    P_ix = find(P2.DATE >= min(P1.DATE) & ...
            P2.DATE <= max(P1.DATE));

    % Shortcut for PLUTO's data
    P_F = P2.FREQUENCY_TX(P_ix);
    P_F_UNIQUE = unique(P_F,'sorted');
    P_DATE = P2.DATE(P_ix);
    P_MOD = P2.MOD(P_ix);
    P_PHASE = P2.PHASE(P_ix);

    % Shortcut for FESTO's data
    F_X = P1.X;
    F_Y = P1.Y;
    F_DATE = P1.DATE;

    FFMOD = cell(1,length(P_F_UNIQUE));
    FFPHASE = cell(1,length(P_F_UNIQUE));
    P_X = cell(1,length(P_F_UNIQUE));
    P_Y = cell(1,length(P_F_UNIQUE));
    if nargout > 3
        P_I = cell(1,length(P_F_UNIQUE));
        P_Q = cell(1,length(P_F_UNIQUE));
    end

    % Cycle on PLUTO's frequencies
    for n=1:length(P_F_UNIQUE)
        fx = find(P_F == P_F_UNIQUE(n));

        % Find the X,Y position of PLUTO's data for current frequency
        P_X{n} = interp1(F_DATE,F_X,P_DATE(fx), 'linear',NaN);
        P_Y{n} = interp1(F_DATE,F_Y,P_DATE(fx), 'linear',NaN);
        
        % Interpolate PLUTO's data over the legacy meshgrid
        FFMOD{n} = scatteredInterpolant(P_X{n},P_Y{n},P_MOD(fx),INTM,NEIG);
        FFPHASE{n} = scatteredInterpolant(P_X{n},P_Y{n},P_PHASE(fx),INTM,NEIG);

        if nargout > 3
            P_I{n} = P_MOD(fx);
            P_Q{n} = P_PHASE(fx);
        end
    end
    end
    P1 = loadFestoLog(FFESTO, H);

    P2 = loadPlutoLog(FPLUTO);
    P2.DATE = P2.DATE + seconds(TOFF*1e-3);

    if any(abs(P2.MOD_I.*sin(P2.PHASE_I)) >= 2^11-1 | ...
            abs(P2.MOD_I.*cos(P2.PHASE_I)) >= 2^11-1)
        disp('WARNING: signal I is saturated.');
    end
    if any(abs(P2.MOD_Q.*sin(P2.PHASE_Q)) >= 2^11-1 | ...
            abs(P2.MOD_Q.*cos(P2.PHASE_Q)) >= 2^11-1)
        disp('WARNING: signal Q is saturated.');
    end

    % Index of PLUTO's data within FESTO's time window
    P_ix = find(P2.DATE >= min(P1.DATE) & ...
            P2.DATE <= max(P1.DATE));

    % Shortcut for PLUTO's data
    P_F = P2.FREQUENCY_TX(P_ix);
    P_F_UNIQUE = unique(P_F,'sorted');
    P_DATE = P2.DATE(P_ix);
    P_MOD = P2.MOD(P_ix);
    P_PHASE = P2.PHASE(P_ix);

    % Shortcut for FESTO's data
    F_X = P1.X;
    F_Y = P1.Y;
    F_DATE = P1.DATE;

    FFMOD = cell(1,length(P_F_UNIQUE));
    FFPHASE = cell(1,length(P_F_UNIQUE));
    P_X = cell(1,length(P_F_UNIQUE));
    P_Y = cell(1,length(P_F_UNIQUE));
    if nargout > 3
        P_I = cell(1,length(P_F_UNIQUE));
        P_Q = cell(1,length(P_F_UNIQUE));
    end

    % Cycle on PLUTO's frequencies
    for n=1:length(P_F_UNIQUE)
        fx = find(P_F == P_F_UNIQUE(n));

        % Find the X,Y position of PLUTO's data for current frequency
        P_X{n} = interp1(F_DATE,F_X,P_DATE(fx), 'linear',NaN);
        P_Y{n} = interp1(F_DATE,F_Y,P_DATE(fx), 'linear',NaN);
        
        % Interpolate PLUTO's data over the legacy meshgrid
        FFMOD{n} = scatteredInterpolant(P_X{n},P_Y{n},P_MOD(fx),INTM,NEIG);
        FFPHASE{n} = scatteredInterpolant(P_X{n},P_Y{n},P_PHASE(fx),INTM,NEIG);

        if nargout > 3
            P_I{n} = P_MOD(fx);
            P_Q{n} = P_PHASE(fx);
        end
    end

end