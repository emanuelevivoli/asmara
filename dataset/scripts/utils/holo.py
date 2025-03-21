import numpy as np

def create_inversion(img,
                    MEDIUM_INDEX = None,
                    WAVELENGTH = 15,
                    SPACING = 0.5,
                    distance = 30,
                    zsteps = 40):
    #TODO: holopy seems to not be available for apple arm chipset. Verify on linux and apple intel chipset
    import holopy as hp

    assert MEDIUM_INDEX is not None, "MEDIUM_INDEX is not defined"

    # load image from file
    raw_holo = hp.load_image(img, 
                            medium_index=MEDIUM_INDEX, 
                            illum_wavelen=WAVELENGTH,
                            illum_polarization=(1,0), 
                            spacing=SPACING)

    zstack = np.linspace(0, distance, zsteps+1)
    rec_vol = hp.propagate(raw_holo, zstack)

    return rec_vol
