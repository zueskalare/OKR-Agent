"""
Chemical Description model implementation.

This class represents chemical components by formula, SMILES strings, or 
other chemical identifiers for constructing molecular systems.
"""
import re
from typing import Dict, Any, Optional


class ChemicalDescription:
    """
    ChemicalDescription defines a chemical component for molecular simulations.
    
    This class stores:
    1. Chemical formula
    2. SMILES representation
    3. Count/stoichiometry
    4. Configuration options
    """
    
    def __init__(
        self,
        formula: Optional[str] = None,
        smiles: Optional[str] = None,
        count: int = 1,
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        inchi: Optional[str] = None
    ):
        """
        Initialize ChemicalDescription.
        
        Args:
            formula: Chemical formula (e.g., 'H2O', 'CH3COOH')
            smiles: SMILES string representation
            count: Number of molecules/formula units
            config: Additional configuration options
            name: Common or IUPAC name of the compound
            inchi: InChI identifier
        """
        self.formula = formula
        self.smiles = smiles
        self.count = int(count)
        self.config = config or {}
        self.name = name
        self.inchi = inchi
        
        # Validate parameters
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate the chemical description.
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        # At least one identifier must be provided
        if not any([self.formula, self.smiles, self.name, self.inchi]):
            raise ValueError("At least one chemical identifier must be provided")
        
        # Count must be positive
        if self.count <= 0:
            raise ValueError("Count must be positive")
        
        # Validate chemical formula if provided
        if self.formula:
            self._validate_formula()
        
        # Validate SMILES if provided
        if self.smiles:
            self._validate_smiles()
            
        return True
    
    def _validate_formula(self):
        """
        Validate chemical formula syntax.
        
        Raises:
            ValueError: If formula is invalid
        """
        # Basic validation of chemical formula
        if not re.match(r'^([A-Z][a-z]?\d*)+$', self.formula):
            raise ValueError(f"Invalid chemical formula: {self.formula}")
    
    def _validate_smiles(self):
        """
        Validate SMILES string syntax.
        
        Raises:
            ValueError: If SMILES is invalid
        """
        # Basic validation only - full validation would require a chemistry library
        if not self.smiles or not isinstance(self.smiles, str):
            raise ValueError("SMILES must be a non-empty string")
            
        # Check for some common invalid characters in SMILES
        invalid_chars = set(' "\'\\')
        if any(c in invalid_chars for c in self.smiles):
            raise ValueError(f"Invalid SMILES string: {self.smiles}")
    
    def get_molecular_weight(self) -> Optional[float]:
        """
        Calculate approximate molecular weight based on formula.
        
        Returns:
            Molecular weight in g/mol or None if formula is not provided
        """
        if not self.formula:
            return None
            
        # Dictionary of common atomic weights
        atomic_weights = {
            'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.811, 
            'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180, 
            'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974, 
            'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078, 
            'Fe': 55.845, 'Cu': 63.546, 'Zn': 65.38, 'Br': 79.904, 'I': 126.904
        }
        
        # Tokenize the formula and calculate weight
        weight = 0.0
        formula = self.formula
        i = 0
        while i < len(formula):
            # Extract element symbol (1 or 2 characters)
            if i + 1 < len(formula) and formula[i].isupper() and formula[i+1].islower():
                element = formula[i:i+2]
                i += 2
            else:
                element = formula[i]
                i += 1
            
            # Extract count (may be multiple digits)
            count = ""
            while i < len(formula) and formula[i].isdigit():
                count += formula[i]
                i += 1
            
            # Default count is 1 if not specified
            count = int(count) if count else 1
            
            # Add to molecular weight
            if element in atomic_weights:
                weight += atomic_weights[element] * count
            else:
                # Unknown element, return None
                return None
                
        return weight
    
    def get_atoms_count(self) -> Optional[Dict[str, int]]:
        """
        Calculate count of each atom type in the formula.
        
        Returns:
            Dictionary mapping element symbols to counts, or None if formula is not provided
        """
        if not self.formula:
            return None
            
        # Parse formula to extract elements and counts
        atoms_count = {}
        formula = self.formula
        i = 0
        while i < len(formula):
            # Extract element symbol (1 or 2 characters)
            if i + 1 < len(formula) and formula[i].isupper() and formula[i+1].islower():
                element = formula[i:i+2]
                i += 2
            else:
                element = formula[i]
                i += 1
            
            # Extract count (may be multiple digits)
            count_str = ""
            while i < len(formula) and formula[i].isdigit():
                count_str += formula[i]
                i += 1
            
            # Default count is 1 if not specified
            count = int(count_str) if count_str else 1
            
            # Update atoms count
            if element in atoms_count:
                atoms_count[element] += count
            else:
                atoms_count[element] = count
                
        return atoms_count
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert chemical description to a dictionary.
        
        Returns:
            Dictionary representation
        """
        result = {
            "count": self.count,
            "config": self.config.copy()
        }
        
        # Include identifiers if they exist
        if self.formula:
            result["formula"] = self.formula
        if self.smiles:
            result["smiles"] = self.smiles
        if self.name:
            result["name"] = self.name
        if self.inchi:
            result["inchi"] = self.inchi
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChemicalDescription':
        """
        Create chemical description from a dictionary.
        
        Args:
            data: Dictionary with description values
            
        Returns:
            ChemicalDescription instance
        """
        return cls(
            formula=data.get("formula"),
            smiles=data.get("smiles"),
            count=data.get("count", 1),
            config=data.get("config", {}),
            name=data.get("name"),
            inchi=data.get("inchi")
        )
    
    def __repr__(self) -> str:
        """String representation of the chemical description."""
        identifiers = []
        if self.formula:
            identifiers.append(f"formula={self.formula}")
        if self.smiles:
            identifiers.append(f"smiles={self.smiles}")
        if self.name:
            identifiers.append(f"name={self.name}")
            
        desc = ", ".join(identifiers)
        if self.count > 1:
            return f"ChemicalDescription({desc}, count={self.count})"
        return f"ChemicalDescription({desc})"
