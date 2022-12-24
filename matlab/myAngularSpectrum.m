function [Hr] = myAngularSpectrum(  H,...               % hologram
                                    d,...               % reconstruction_distance
                                    w,...               % wavelength
                                    dx,...              % pixel_size
                                    B,...               % phase_mask
                                    scale_factor)

    %%%%%Begin Function
    A = size(H);

    if nargin < 6
        scale_factor = 1;   
    end
    if nargin < 5
        B = 1; 
    end
    if nargin < 4
        error('Not enough input arguments')    
    end
    if length(B) == 1
        B = ones(A).*B; 
    end

    %use double precision to allow for complex numbers
    H = double(H);
    B = double(B);

    n = length(H);   %size of hologram matrix nxn
    H = double(H)-mean(mean(H));  %must be class double for imaginary #s

    %%%%%%Angular Spectrum Method
    k0=2*pi/w;              %k-vector
    k = (-n/2:1:n/2-1)*dx;  %array same dimentions as hologram
    l = (-n/2:1:n/2-1)*dx;
    [XX,YY] = meshgrid(k,l);
    step = k(2)-k(1);  %step size

    k_max = 2*pi/step;  %maximum k vector
    k_axis = linspace(0,k_max,n)-k_max/2;  %confine reconstruction to k<k_max 
    [Kx Ky]=meshgrid(k_axis,k_axis);

    E=B.*exp(-1i*sqrt(4.*k0^2-Kx.^2-Ky.^2)*d); %Angular spectrum
    Hf=fftshift(fft2(sqrt(H)));
    [Hr]=ifft2((Hf.*E));    
end %end function
    
 


