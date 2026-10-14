"""
Cell model implementation.

This class represents a unit cell or simulation box with
dimensions and periodicity information.
"""
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class Cell:
    """
    Cell represents a unit cell or simulation box.
    
    This class stores:
    1. Cell dimensions (a, b, c)
    2. Cell angles (alpha, beta, gamma)
    3. Methods to compute cell properties and transformations
    """
    
    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        alpha: float = 90.0,
        beta: float = 90.0,
        gamma: float = 90.0,
        pbc: Optional[List[bool]] = None
    ):
        """
        Initialize a Cell.
        
        Args:
            a: Length of a-axis (Å)
            b: Length of b-axis (Å)
            c: Length of c-axis (Å)
            alpha: Angle between b and c axes (degrees)
            beta: Angle between a and c axes (degrees)
            gamma: Angle between a and b axes (degrees)
            pbc: Periodic boundary conditions [x, y, z] (default: all True)
        """
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.pbc = pbc if pbc is not None else [True, True, True]
        
        # Cache cell vectors
        self._vectors = None
        self._volume = None
    
    def get_volume(self) -> float:
        """
        Calculate the volume of the cell.
        
        Returns:
            Cell volume in Å³
        """
        if self._volume is None:
            # Convert angles to radians
            alpha_rad = math.radians(self.alpha)
            beta_rad = math.radians(self.beta)
            gamma_rad = math.radians(self.gamma)
            
            # Calculate volume
            term = (1 - math.cos(alpha_rad)**2 - math.cos(beta_rad)**2 - math.cos(gamma_rad)**2 
                   + 2 * math.cos(alpha_rad) * math.cos(beta_rad) * math.cos(gamma_rad))
            
            self._volume = self.a * self.b * self.c * math.sqrt(max(0.0, term))
        
        return self._volume
    
    def get_vectors(self) -> Tuple[Tuple[float, float, float], ...]:
        """
        Get the cell vectors.
        
        Returns:
            Tuple of 3 vectors (a, b, c)
        """
        if self._vectors is None:
            # Convert angles to radians
            alpha_rad = math.radians(self.alpha)
            beta_rad = math.radians(self.beta)
            gamma_rad = math.radians(self.gamma)
            
            # First vector along x-axis
            ax = self.a
            ay = 0.0
            az = 0.0
            
            # Second vector in xy-plane
            bx = self.b * math.cos(gamma_rad)
            by = self.b * math.sin(gamma_rad)
            bz = 0.0
            
            # Third vector with components in all directions
            cx = self.c * math.cos(beta_rad)
            cy = (self.c * math.cos(alpha_rad) - cx * math.cos(gamma_rad)) / math.sin(gamma_rad)
            
            # z-component from volume constraint
            term = 1.0 - math.cos(alpha_rad)**2 - math.cos(beta_rad)**2 - math.cos(gamma_rad)**2
            term += 2 * math.cos(alpha_rad) * math.cos(beta_rad) * math.cos(gamma_rad)
            
            cz_sq = self.c**2 - cx**2 - cy**2
            cz = math.sqrt(max(0.0, cz_sq))
            
            self._vectors = ((ax, ay, az), (bx, by, bz), (cx, cy, cz))
        
        return self._vectors
    
    def to_orthorhombic(self) -> 'Cell':
        """
        Convert to an orthorhombic cell (all angles 90°).
        
        Returns:
            New orthorhombic Cell instance
        """
        # If already orthorhombic (or close enough), return a copy
        if (abs(self.alpha - 90.0) < 1e-6 and 
            abs(self.beta - 90.0) < 1e-6 and 
            abs(self.gamma - 90.0) < 1e-6):
            return Cell(self.a, self.b, self.c)
        
        # Get cell volume and create a cubic cell with the same volume
        volume = self.get_volume()
        side_length = volume ** (1/3)
        
        return Cell(side_length, side_length, side_length)
    
    def fractional_to_cartesian(self, frac_coords: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Convert fractional coordinates to Cartesian coordinates.
        
        Args:
            frac_coords: Fractional coordinates (fx, fy, fz)
            
        Returns:
            Cartesian coordinates (x, y, z)
        """
        fx, fy, fz = frac_coords
        vectors = self.get_vectors()
        
        x = fx * vectors[0][0] + fy * vectors[1][0] + fz * vectors[2][0]
        y = fx * vectors[0][1] + fy * vectors[1][1] + fz * vectors[2][1]
        z = fx * vectors[0][2] + fy * vectors[1][2] + fz * vectors[2][2]
        
        return (x, y, z)
    
    def cartesian_to_fractional(self, coords: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Convert Cartesian coordinates to fractional coordinates.
        
        Args:
            coords: Cartesian coordinates (x, y, z)
            
        Returns:
            Fractional coordinates (fx, fy, fz)
        """
        # This is an approximation for a general cell
        # A real implementation would use matrix inversion
        
        # Get cell vectors
        vectors = self.get_vectors()
        
        # Convert to numpy arrays for easier calculation
        a_vec = np.array(vectors[0])
        b_vec = np.array(vectors[1])
        c_vec = np.array(vectors[2])
        coords_vec = np.array(coords)
        
        # Compute cell matrix
        cell_matrix = np.vstack((a_vec, b_vec, c_vec)).T
        
        # Inverse of cell matrix
        inv_cell = np.linalg.inv(cell_matrix)
        
        # Compute fractional coordinates
        frac = np.dot(inv_cell, coords_vec)
        
        return tuple(frac)
    
    def wrap_coordinates(self, coords: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Wrap coordinates according to periodic boundary conditions.
        
        Args:
            coords: Cartesian coordinates (x, y, z)
            
        Returns:
            Wrapped Cartesian coordinates (x', y', z')
        """
        # Convert to fractional coordinates
        frac = self.cartesian_to_fractional(coords)
        
        # Wrap fractional coordinates
        wrapped_frac = []
        for i, f in enumerate(frac):
            if self.pbc[i]:
                wrapped_frac.append(f - math.floor(f))
            else:
                wrapped_frac.append(f)
        
        # Convert back to Cartesian
        return self.fractional_to_cartesian(tuple(wrapped_frac))
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the cell to a dictionary representation.
        
        Returns:
            Dictionary with cell properties
        """
        return {
            'a': self.a,
            'b': self.b,
            'c': self.c,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'pbc': self.pbc.copy(),
            'volume': self.get_volume()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Cell':
        """
        Create a Cell instance from a dictionary.
        
        Args:
            data: Dictionary with cell properties
            
        Returns:
            Cell instance
        """
        return cls(
            a=data['a'],
            b=data['b'],
            c=data['c'],
            alpha=data.get('alpha', 90.0),
            beta=data.get('beta', 90.0),
            gamma=data.get('gamma', 90.0),
            pbc=data.get('pbc', [True, True, True])
        )
    
    @classmethod
    def cubic(cls, side_length: float, pbc: Optional[List[bool]] = None) -> 'Cell':
        """
        Create a cubic cell.
        
        Args:
            side_length: Length of each side
            pbc: Periodic boundary conditions
            
        Returns:
            Cell instance
        """
        return cls(side_length, side_length, side_length, 90.0, 90.0, 90.0, pbc)
    
    def __repr__(self) -> str:
        """String representation of the cell."""
        return f"Cell(a={self.a:.3f}, b={self.b:.3f}, c={self.c:.3f}, α={self.alpha:.1f}°, β={self.beta:.1f}°, γ={self.gamma:.1f}°)"
