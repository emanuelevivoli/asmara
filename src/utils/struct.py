import numpy as np
import torch
import torch.nn.functional as F

# define a class for holograms
class Holo:
    """Class for holograms.
    
    Attributes:
        hologram (np.ndarray): hologram as a numpy array
    """
    # get as input a hologram as a numpy array
    def __init__(self, hologram: np.ndarray) -> None:
        self.hologram = hologram
    
    # define the sum operation as the addition of the holograms
    def __add__(self, other):
        # check shape
        if self.hologram.shape != other.hologram.shape:
            bigger_self = self.hologram.shape > other.hologram.shape
            
            if bigger_self: other.hologram = self.reshape(other.hologram, self.hologram.shape)
            else: self.hologram = self.reshape(self.hologram, other.hologram.shape)
                       
        return self.hologram + other.hologram
    
    def __mul__(self, scalar):
        """Multiply the hologram by a scalar and return a new Holo object."""
        if not isinstance(scalar, (int, float)):
            raise TypeError("Multiplication only supports int or float scalar types")
        new_hologram = self.hologram * scalar
        return Holo(new_hologram)

    def __rmul__(self, scalar):
        """Handle multiplication when Holo is the right operand."""
        return self.__mul__(scalar)
    
    # define the save operation for a generic Hologram, given a path
    def save(self, path):
        np.save(path, self.hologram)
        return self
    
    # define the reshape operation for a generic Hologram, given a shape
    def reshape(self, hologram, shape):
        hologram = np.resize(hologram, shape)
        return hologram
    
    def interpolate(self, size=(60, 60)):
        torch_holo = torch.from_numpy(self.hologram)
                
        # get real and imaginary part of the hologram
        x_real = torch_holo.real.unsqueeze(0).unsqueeze(0)
        x_imag = torch_holo.imag.unsqueeze(0).unsqueeze(0)
                
        # interpolate the hologram
        rescaled_real = F.interpolate(x_real, size=size, mode='bilinear', align_corners=False)
        rescaled_imag = F.interpolate(x_imag, size=size, mode='bilinear', align_corners=False)
                
        # fuse real and imaginary part
        torch_holo = torch.complex(rescaled_real, rescaled_imag).squeeze(0).squeeze(0)
        self.hologram = torch_holo.numpy()
        return self

# there is no need to define a class for images

# define a class for holograms Inversions
class HoloInv:
    """Class for holograms Inversions.
    
    Attributes:
        hologram (np.ndarray): hologram as a numpy array
    """
    # get as input a hologram as a numpy multi-dimensional array
    def __init__(self, hologram: np.ndarray) -> None:
        self.hologram = hologram

    # define the sum operation as the addition of the holograms
    def __add__(self, other):
        """Add two holograms.
        
        Args:
            other (HoloInv): other hologram to add
            
        Returns:
            np.ndarray: sum of the two holograms
            
        Examples:
            >>> holo1 = HoloInv(np.array([1, 2, 3]))
            >>> holo2 = HoloInv(np.array([4, 5, 6]))
            >>> holo1 + holo2
            array([5, 7, 9])
        """
        return self.hologram + other.hologram

