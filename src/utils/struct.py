import numpy as np

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
    
    # define the reshape operation for a generic Hologram, given a shape
    def reshape(self, hologram, shape):
        hologram = np.resize(hologram, shape)
        return hologram

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

