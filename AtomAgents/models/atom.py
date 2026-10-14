"""
Atom model implementation.

This class represents an atom in a molecular structure with
coordinates, element type, and bonding information.
"""
from typing import List, Dict, Any, Optional


class Atom:
    """
    Atom represents a single atom in a molecular structure.
    
    This class stores:
    1. Elemental information (element type, atomic number)
    2. Spatial coordinates (x, y, z)
    3. Connectivity (bonds to other atoms)
    4. Force field parameters (charge, atom type)
    """
    
    def __init__(
        self, 
        element: str, 
        x: float, 
        y: float, 
        z: float, 
        id: Optional[int] = None,
        atom_type: Optional[str] = None,
        charge: float = 0.0,
        mass: Optional[float] = None,
        bonds: Optional[List[int]] = None,
        properties: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an Atom.
        
        Args:
            element: Chemical element symbol (e.g., 'C', 'H', 'O')
            x: X-coordinate (Å)
            y: Y-coordinate (Å)
            z: Z-coordinate (Å)
            id: Atom ID or index (optional)
            atom_type: Force field atom type (optional)
            charge: Partial charge (e)
            mass: Atomic mass (amu, optional)
            bonds: List of atom IDs this atom is bonded to (optional)
            properties: Additional atom properties (optional)
        """
        self.element = element
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.id = id
        self.atom_type = atom_type
        self.charge = float(charge)
        self._mass = mass
        self.bonds = bonds or []
        self.properties = properties or {}

    @property
    def mass(self) -> float:
        """
        Get the atomic mass. If not explicitly set, returns the standard
        mass for the element.
        
        Returns:
            Atomic mass in amu
        """
        if self._mass is not None:
            return self._mass
        
        # Return standard mass for common elements
        # This is a simplified approach - a real implementation would use a complete periodic table
        element_masses = {
            'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'P': 30.974,
            'S': 32.065, 'F': 18.998, 'Cl': 35.453, 'Br': 79.904, 'I': 126.904,
            'Na': 22.990, 'K': 39.098, 'Ca': 40.078, 'Mg': 24.305, 'Fe': 55.845
        }
        
        return element_masses.get(self.element, 0.0)
    
    @mass.setter
    def mass(self, value: float):
        """Set the atomic mass."""
        self._mass = float(value)
    
    def get_position(self) -> tuple[float, float, float]:
        """
        Get the atomic coordinates.
        
        Returns:
            Tuple of (x, y, z) coordinates
        """
        return (self.x, self.y, self.z)
    
    def set_position(self, x: float, y: float, z: float):
        """
        Set the atomic coordinates.
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            z: Z-coordinate
        """
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    def add_bond(self, atom_id: int):
        """
        Add a bond to another atom.
        
        Args:
            atom_id: ID of the atom to bond to
        """
        if atom_id not in self.bonds:
            self.bonds.append(atom_id)
    
    def remove_bond(self, atom_id: int):
        """
        Remove a bond to another atom.
        
        Args:
            atom_id: ID of the atom to remove bond from
        """
        if atom_id in self.bonds:
            self.bonds.remove(atom_id)
    
    def distance_to(self, other: 'Atom') -> float:
        """
        Calculate distance to another atom.
        
        Args:
            other: The other atom
            
        Returns:
            Distance in Å
        """
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return (dx*dx + dy*dy + dz*dz) ** 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the atom to a dictionary representation.
        
        Returns:
            Dictionary with atom properties
        """
        return {
            'element': self.element,
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'id': self.id,
            'atom_type': self.atom_type,
            'charge': self.charge,
            'mass': self.mass,
            'bonds': self.bonds.copy(),
            'properties': self.properties.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Atom':
        """
        Create an Atom instance from a dictionary.
        
        Args:
            data: Dictionary with atom properties
            
        Returns:
            Atom instance
        """
        return cls(
            element=data['element'],
            x=data['x'],
            y=data['y'],
            z=data['z'],
            id=data.get('id'),
            atom_type=data.get('atom_type'),
            charge=data.get('charge', 0.0),
            mass=data.get('mass'),
            bonds=data.get('bonds', []),
            properties=data.get('properties', {})
        )
    
    def __repr__(self) -> str:
        """String representation of the atom."""
        return f"Atom({self.element}, id={self.id}, pos=({self.x:.3f}, {self.y:.3f}, {self.z:.3f}))"
