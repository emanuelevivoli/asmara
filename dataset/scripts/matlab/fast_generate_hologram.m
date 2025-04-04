function [MO, PH, H, Hfill] = fast_generate_hologram(F, FI, FQ, STEP)

% Set default values
if ~exist('STEP','var') || isempty(STEP)
    STEP = 5;
end

for n=1:length(F)
    
    [X,Y] = meshgrid(min(FI{1}.Points(:,1)):STEP:max(FI{1}.Points(:,1)), ...
        min(FI{1}.Points(:,2)):STEP:max(FI{1}.Points(:,2)));
    
    % original values
    MO = FI{n}(X,Y);
    PH = FQ{n}(X,Y);
    H = MO.*exp(1i.*PH);
    
    % values after fill-outliers
    MO = filloutliers(MO,'pchip','gesd');
    PH = filloutliers(PH,'pchip','gesd');
    Hfill = MO.*exp(1i.*PH);

end
