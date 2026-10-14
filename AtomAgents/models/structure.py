"""
Structure model implementation.

This class represents a molecular structure with atoms, bonds,
and spatial arrangement within a cell or simulation box.
"""
import os
import json
from typing import List, Dict, Any, Optional, Tuple

from .atom import Atom
from .cell import Cell


class Structure:
    """
    Structure represents a molecular structure with atoms in 3D space.
    
    This class stores:
    1. Collection of atoms with their properties
    2. Simulation cell or unit cell information
    3. Methods for structure manipulation and analysis
    4. Metadata about the structure
    """
    
    def __init__(
        self,
        atoms: Optional[List[Atom]] = None,
        cell: Optional[Cell] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a Structure.
        
        Args:
            atoms: List of Atom objects (optional)
            cell: Cell object defining the simulation box (optional)
            metadata: Additional structure metadata (optional)
        """
        self.atoms = atoms or []
        
        # If no cell is provided, create a default one based on atoms
        if cell is None and self.atoms:
            self.cell = self._create_default_cell()
        else:
            self.cell = cell or Cell(10.0, 10.0, 10.0)  # Default 10Å box
        
        self.metadata = metadata or {}
        
        # Make sure atom IDs are set
        self._ensure_atom_ids()
    
    def _ensure_atom_ids(self):
        """Ensure all atoms have unique IDs."""
        for i, atom in enumerate(self.atoms):
            if atom.id is None:
                atom.id = i
    
    def _create_default_cell(self) -> Cell:
        """
        Create a default cell based on atom positions.
        
        Returns:
            Cell object with dimensions based on atom positions
        """
        if not self.atoms:
            return Cell(10.0, 10.0, 10.0)  # Default 10Å box
        
        # Get min and max coordinates
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        
        for atom in self.atoms:
            min_x = min(min_x, atom.x)
            min_y = min(min_y, atom.y)
            min_z = min(min_z, atom.z)
            max_x = max(max_x, atom.x)
            max_y = max(max_y, atom.y)
            max_z = max(max_z, atom.z)
        
        # Add padding
        padding = 5.0  # 5Å padding
        a = max(max_x - min_x + 2 * padding, 1.0)
        b = max(max_y - min_y + 2 * padding, 1.0)
        c = max(max_z - min_z + 2 * padding, 1.0)
        
        return Cell(a, b, c)
    
    def add_atom(self, atom: Atom):
        """
        Add an atom to the structure.
        
        Args:
            atom: Atom object to add
        """
        # Set ID if not already set
        if atom.id is None:
            atom.id = len(self.atoms)
        
        self.atoms.append(atom)
    
    def remove_atom(self, atom_id: int):
        """
        Remove an atom from the structure.
        
        Args:
            atom_id: ID of the atom to remove
        """
        # Find the atom with the given ID
        for i, atom in enumerate(self.atoms):
            if atom.id == atom_id:
                # Remove the atom
                del self.atoms[i]
                
                # Update bonds in other atoms
                for a in self.atoms:
                    if atom_id in a.bonds:
                        a.bonds.remove(atom_id)
                
                return
        
        raise ValueError(f"Atom with ID {atom_id} not found")
    
    def get_atom(self, atom_id: int) -> Optional[Atom]:
        """
        Get an atom by ID.
        
        Args:
            atom_id: ID of the atom
            
        Returns:
            Atom object or None if not found
        """
        for atom in self.atoms:
            if atom.id == atom_id:
                return atom
        return None
    
    def get_bonds(self) -> List[Tuple[int, int]]:
        """
        Get all bonds in the structure.
        
        Returns:
            List of (atom_id1, atom_id2) tuples representing bonds
        """
        bonds = set()
        
        for atom in self.atoms:
            atom_id = atom.id
            for bonded_id in atom.bonds:
                # Ensure we don't count the same bond twice
                bond = tuple(sorted([atom_id, bonded_id]))
                bonds.add(bond)
        
        return list(bonds)
    
    def compute_center_of_mass(self) -> Tuple[float, float, float]:
        """
        Compute the center of mass of the structure.
        
        Returns:
            Tuple (x, y, z) representing the center of mass
        """
        if not self.atoms:
            return (0.0, 0.0, 0.0)
        
        total_mass = 0.0
        weighted_x = weighted_y = weighted_z = 0.0
        
        for atom in self.atoms:
            mass = atom.mass
            total_mass += mass
            weighted_x += mass * atom.x
            weighted_y += mass * atom.y
            weighted_z += mass * atom.z
        
        if total_mass > 0:
            return (weighted_x / total_mass, weighted_y / total_mass, weighted_z / total_mass)
        else:
            return (0.0, 0.0, 0.0)
    
    def translate(self, dx: float, dy: float, dz: float):
        """
        Translate all atoms.
        
        Args:
            dx: Translation along x-axis
            dy: Translation along y-axis
            dz: Translation along z-axis
        """
        for atom in self.atoms:
            atom.x += dx
            atom.y += dy
            atom.z += dz
    
    def center(self):
        """
        Center the structure in the cell.
        """
        # Compute current center of mass
        com_x, com_y, com_z = self.compute_center_of_mass()
        
        # Compute cell center
        cell_center_x = self.cell.a / 2
        cell_center_y = self.cell.b / 2
        cell_center_z = self.cell.c / 2
        
        # Compute translation
        dx = cell_center_x - com_x
        dy = cell_center_y - com_y
        dz = cell_center_z - com_z
        
        # Apply translation
        self.translate(dx, dy, dz)
    
    def wrap_to_cell(self):
        """
        Wrap atoms to the simulation cell.
        """
        for atom in self.atoms:
            coords = (atom.x, atom.y, atom.z)
            wrapped_coords = self.cell.wrap_coordinates(coords)
            atom.x, atom.y, atom.z = wrapped_coords
    
    def guess_bonds(self, tolerance: float = 0.45):
        """
        Guess bonds based on interatomic distances.
        
        Args:
            tolerance: Tolerance factor for bond detection
        """
        # Clear existing bonds
        for atom in self.atoms:
            atom.bonds = []
        
        # Covalent radii for common elements (Å)
        covalent_radii = {
            'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
            'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39,
            'Na': 1.66, 'K': 2.03, 'Mg': 1.41, 'Ca': 1.76, 'Fe': 1.32
        }
        
        # Check all pairs of atoms
        for i, atom1 in enumerate(self.atoms):
            for j, atom2 in enumerate(self.atoms):
                if i >= j:
                    continue  # Skip self and already processed pairs
                
                # Get covalent radii
                r1 = covalent_radii.get(atom1.element, 0.7)
                r2 = covalent_radii.get(atom2.element, 0.7)
                
                # Compute distance
                dist = atom1.distance_to(atom2)
                
                # Check if atoms are within bonding distance
                bond_length = r1 + r2
                if dist < bond_length * (1.0 + tolerance):
                    atom1.add_bond(atom2.id)
                    atom2.add_bond(atom1.id)
    
    def to_file(self, format: str, filename: str):
        """
        Write the structure to a file.
        
        Args:
            format: File format ('xyz', 'pdb', 'json')
            filename: Output filename
        """
        if format.lower() == 'xyz':
            self._write_xyz(filename)
        elif format.lower() == 'pdb':
            self._write_pdb(filename)
        elif format.lower() == 'json':
            self._write_json(filename)
        else:
            raise ValueError(f"Unsupported file format: {format}")
    
    def _write_xyz(self, filename: str):
        """Write structure in XYZ format."""
        with open(filename, 'w') as f:
            # Write number of atoms
            f.write(f"{len(self.atoms)}\n")
            
            # Write comment line with cell info
            cell_info = f"Cell: {self.cell.a:.3f} {self.cell.b:.3f} {self.cell.c:.3f} "
            cell_info += f"{self.cell.alpha:.1f} {self.cell.beta:.1f} {self.cell.gamma:.1f}"
            f.write(f"{cell_info}\n")
            
            # Write atoms
            for atom in self.atoms:
                f.write(f"{atom.element} {atom.x:.6f} {atom.y:.6f} {atom.z:.6f}\n")
    
    def _write_pdb(self, filename: str):
        """Write structure in PDB format."""
        with open(filename, 'w') as f:
            # Write header
            f.write("REMARK Generated by AtomAgent\n")
            
            # Write cell information
            f.write(f"CRYST1{self.cell.a:9.3f}{self.cell.b:9.3f}{self.cell.c:9.3f}")
            f.write(f"{self.cell.alpha:7.2f}{self.cell.beta:7.2f}{self.cell.gamma:7.2f}")
            f.write(" P 1           1\n")
            
            # Write atoms
            for i, atom in enumerate(self.atoms):
                atom_id = atom.id if atom.id is not None else i + 1
                x, y, z = atom.x, atom.y, atom.z
                
                line = f"ATOM  {atom_id:5d} {atom.element:<4s}{'':1s}{'':3s} {'':1s}{1:4d}{'':1s}   "
                line += f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}      "
                line += f"{'':4s}{atom.element:>2s}{'':2s}\n"
                
                f.write(line)
            
            # Write bonds as CONECT records
            for atom in self.atoms:
                if atom.bonds:
                    line = f"CONECT{atom.id:5d}"
                    for bonded_id in atom.bonds:
                        line += f"{bonded_id:5d}"
                    line += "\n"
                    f.write(line)
            
            # Write end
            f.write("END\n")
    
    def _write_json(self, filename: str):
        """Write structure in JSON format."""
        data = {
            'atoms': [atom.to_dict() for atom in self.atoms],
            'cell': self.cell.to_dict(),
            'metadata': self.metadata
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def from_file(cls, filename: str, format: Optional[str] = None) -> 'Structure':
        """
        Read a structure from a file.
        
        Args:
            filename: Input filename
            format: File format (if None, guess from extension)
            
        Returns:
            Structure object
        """
        # Determine format if not specified
        if format is None:
            _, ext = os.path.splitext(filename)
            format = ext[1:]  # Remove the dot
        
        format = format.lower()
        
        if format == 'xyz':
            return cls._read_xyz(filename)
        elif format == 'pdb':
            return cls._read_pdb(filename)
        elif format in ('json', 'jsn'):
            return cls._read_json(filename)
        else:
            raise ValueError(f"Unsupported file format: {format}")
    
    @classmethod
    def _read_xyz(cls, filename: str) -> 'Structure':
        """Read structure from XYZ format."""
        atoms = []
        cell = None
        
        with open(filename, 'r') as f:
            lines = f.readlines()
            
            # Read number of atoms
            n_atoms = int(lines[0].strip())
            
            # Read comment line, which might contain cell info
            comment = lines[1].strip()
            cell_pattern = r"Cell:\s*([\d\.]+)\s*([\d\.]+)\s*([\d\.]+)\s*([\d\.]+)\s*([\d\.]+)\s*([\d\.]+)"
            import re
            cell_match = re.search(cell_pattern, comment)
            if cell_match:
                a, b, c, alpha, beta, gamma = map(float, cell_match.groups())
                cell = Cell(a, b, c, alpha, beta, gamma)
            
            # Read atoms
            for i in range(2, 2 + n_atoms):
                if i >= len(lines):
                    break
                    
                parts = lines[i].strip().split()
                if len(parts) >= 4:
                    element = parts[0]
                    x, y, z = map(float, parts[1:4])
                    atoms.append(Atom(element, x, y, z, id=i-2))
        
        return cls(atoms, cell)
    
    @classmethod
    def _read_pdb(cls, filename: str) -> 'Structure':
        """Read structure from PDB format."""
        atoms = []
        cell = None
        bonds = {}  # Dictionary to store bonds from CONECT records
        
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith("CRYST1"):
                    # Parse cell parameters
                    try:
                        a = float(line[6:15].strip())
                        b = float(line[15:24].strip())
                        c = float(line[24:33].strip())
                        alpha = float(line[33:40].strip())
                        beta = float(line[40:47].strip())
                        gamma = float(line[47:54].strip())
                        cell = Cell(a, b, c, alpha, beta, gamma)
                    except (ValueError, IndexError):
                        # If cell parameters cannot be parsed, use default
                        pass
                
                elif line.startswith("ATOM") or line.startswith("HETATM"):
                    # Parse atom
                    try:
                        atom_id = int(line[6:11].strip())
                        element = line[76:78].strip()
                        if not element:  # If element not in columns 76-78, try to get from atom name
                            element = line[12:16].strip()[0]  # First letter of atom name
                        
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        
                        # Try to extract charge and atom type if available
                        charge = 0.0
                        atom_type = None
                        
                        atoms.append(Atom(element, x, y, z, id=atom_id, charge=charge, atom_type=atom_type))
                    except (ValueError, IndexError):
                        continue
                
                elif line.startswith("CONECT"):
                    # Parse connectivity
                    parts = line.strip().split()[1:]  # Skip "CONECT"
                    if parts:
                        try:
                            atom_id = int(parts[0])
                            bonded_ids = [int(p) for p in parts[1:]]
                            bonds[atom_id] = bonded_ids
                        except ValueError:
                            continue
        
        # Create structure
        structure = cls(atoms, cell)
        
        # Apply bonds
        for atom_id, bonded_ids in bonds.items():
            atom = structure.get_atom(atom_id)
            if atom:
                atom.bonds = bonded_ids
        
        return structure
    
    @classmethod
    def _read_json(cls, filename: str) -> 'Structure':
        """Read structure from JSON format."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Parse cell
        cell_data = data.get('cell', {})
        cell = Cell.from_dict(cell_data) if cell_data else None
        
        # Parse atoms
        atoms_data = data.get('atoms', [])
        atoms = [Atom.from_dict(atom_data) for atom_data in atoms_data]
        
        # Get metadata
        metadata = data.get('metadata', {})
        
        return cls(atoms, cell, metadata)
    
    def copy(self) -> 'Structure':
        """
        Create a deep copy of the structure.
        
        Returns:
            New Structure instance with copied atoms and cell
        """
        # Copy atoms
        new_atoms = []
        for atom in self.atoms:
            new_atom = Atom(
                element=atom.element,
                x=atom.x, 
                y=atom.y, 
                z=atom.z,
                id=atom.id,
                atom_type=atom.atom_type,
                charge=atom.charge,
                mass=atom._mass,  # Directly access _mass to ensure exact copy
                bonds=atom.bonds.copy(),
                properties=atom.properties.copy()
            )
            new_atoms.append(new_atom)
        
        # Copy cell
        new_cell = Cell(
            a=self.cell.a,
            b=self.cell.b,
            c=self.cell.c,
            alpha=self.cell.alpha,
            beta=self.cell.beta,
            gamma=self.cell.gamma,
            pbc=self.cell.pbc.copy()
        )
        
        # Create new structure with copied metadata
        return Structure(
            atoms=new_atoms,
            cell=new_cell,
            metadata=self.metadata.copy()
        )
    
    def __len__(self) -> int:
        """Get the number of atoms in the structure."""
        return len(self.atoms)
    
    def __iter__(self):
        """Iterate over atoms in the structure."""
        return iter(self.atoms)
    
    def __repr__(self) -> str:
        """String representation of the structure."""
        elements = {}
        for atom in self.atoms:
            elements[atom.element] = elements.get(atom.element, 0) + 1
        
        composition = " ".join(f"{el}{count}" for el, count in elements.items())
        return f"Structure(atoms={len(self.atoms)}, composition={composition}, cell={self.cell})"
