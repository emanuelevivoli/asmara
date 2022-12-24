function [Inverted] = fast_inversion (  H,...
                                        F,...
                                        epsilon,... 
                                        choice,...
                                        in_line,...
                                        min,...
                                        max,...
                                        step,...
                                        pad_size,...
                                        scale_factor,...
                                        resizeFlag,...
                                        resizeFactor)

    if nargin < 12
        resizeFactor = double(1);
    end
    if nargin < 11
        resizeFlag = double(0);
    end
    if nargin < 10
        scale_factor = double(1);
    end
    if nargin < 9
        pad_size = double(0);
    end
    if nargin < 8
        step = double(40);
    end
    if nargin < 7
        max = double(20)/1e2;
    end
    if nargin < 6
        min = double(0)/1e2;
    end
    if nargin < 5
    %     in_line = 0;
    % end
    % if nargin < 4
    %     choice = 'Fresnel';
    % end
    % if nargin < 3
        error('Not enough input arguments')    
    end

    lambda = 3e8./(F*sqrt(epsilon));
    HH=imresize(H,[62*resizeFactor 62*resizeFactor]);

    b = min; 
    c = (max-min)/step; 
    d = 1;
    Inverted = cell(1,1,step);

    switch choice
        case 'Fresnel'
            error('Not enough input arguments')
            
        case 'Conv'
            error('Not enough input arguments')
            
        case 'AngSpec'
            error('Not enough input arguments')        

        case 'newAngSpec'
            while d<=step+1    
                Inverted{:,:,d}=myAngularSpectrum(HH,b,lambda,0.005);
                b = b+c;
                d = d+1;
            end   
    end