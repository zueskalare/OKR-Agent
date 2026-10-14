"""
Simulation Parameters model implementation.

This class represents the configuration parameters for molecular dynamics simulations,
including ensemble settings, thermodynamic conditions, and timestepping.
"""
from typing import Dict, Any, Optional


class SimulationParameters:
    """
    SimulationParameters defines the settings for molecular dynamics simulations.
    
    This class stores:
    1. Ensemble configuration (NVT, NPT, etc.)
    2. Thermodynamic conditions (temperature, pressure)
    3. Temporal settings (timestep, run length)
    4. Force field specification
    5. Additional custom settings
    """
    
    def __init__(
        self,
        ensemble: str = "NPT",
        temperature: float = 300.0,
        pressure: float = 1.0,
        timestep: float = 2.0,
        equilibration_time: float = 100.0,
        production_time: float = 1000.0,
        force_field: str = "OPLS",
        custom_settings: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize SimulationParameters.
        
        Args:
            ensemble: Simulation ensemble (e.g., 'NVE', 'NVT', 'NPT')
            temperature: Simulation temperature in Kelvin
            pressure: Simulation pressure in atmospheres
            timestep: Integration timestep in femtoseconds
            equilibration_time: Equilibration phase duration in picoseconds
            production_time: Production phase duration in picoseconds
            force_field: Force field to use for the simulation
            custom_settings: Additional simulation settings
        """
        self.ensemble = ensemble
        self.temperature = float(temperature)
        self.pressure = float(pressure)
        self.timestep = float(timestep)
        self.equilibration_time = float(equilibration_time)
        self.production_time = float(production_time)
        self.force_field = force_field
        self.custom_settings = custom_settings or {}
        
        # Validate parameters
        self._validate()
    
    def _validate(self):
        """Validate parameter values."""
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        
        if self.timestep <= 0:
            raise ValueError("Timestep must be positive")
        
        if self.equilibration_time < 0:
            raise ValueError("Equilibration time cannot be negative")
        
        if self.production_time <= 0:
            raise ValueError("Production time must be positive")
            
        if self.ensemble not in ["NVE", "NVT", "NPT", "μVT"]:
            # Allow but warn about non-standard ensembles
            import warnings
            warnings.warn(f"Non-standard ensemble: {self.ensemble}")
    
    def get_equilibration_steps(self) -> int:
        """
        Calculate number of steps needed for equilibration.
        
        Returns:
            Number of simulation steps
        """
        # Convert from picoseconds to steps
        # 1 ps = 1000 fs
        return int((self.equilibration_time * 1000) / self.timestep)
    
    def get_production_steps(self) -> int:
        """
        Calculate number of steps needed for production run.
        
        Returns:
            Number of simulation steps
        """
        # Convert from picoseconds to steps
        # 1 ps = 1000 fs
        return int((self.production_time * 1000) / self.timestep)
    
    def get_total_time(self) -> float:
        """
        Get total simulation time in picoseconds.
        
        Returns:
            Total time in picoseconds
        """
        return self.equilibration_time + self.production_time
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a custom setting or return default if not found.
        
        Args:
            key: Setting name
            default: Default value if setting not found
            
        Returns:
            Setting value or default
        """
        return self.custom_settings.get(key, default)
    
    def set_setting(self, key: str, value: Any):
        """
        Set a custom setting.
        
        Args:
            key: Setting name
            value: Setting value
        """
        self.custom_settings[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameters to a dictionary.
        
        Returns:
            Dictionary representation of parameters
        """
        return {
            "ensemble": self.ensemble,
            "temperature": self.temperature,
            "pressure": self.pressure,
            "timestep": self.timestep,
            "equilibration_time": self.equilibration_time,
            "production_time": self.production_time,
            "force_field": self.force_field,
            "custom_settings": self.custom_settings.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationParameters':
        """
        Create parameters from a dictionary.
        
        Args:
            data: Dictionary with parameter values
            
        Returns:
            SimulationParameters instance
        """
        return cls(
            ensemble=data.get("ensemble", "NPT"),
            temperature=data.get("temperature", 300.0),
            pressure=data.get("pressure", 1.0),
            timestep=data.get("timestep", 2.0),
            equilibration_time=data.get("equilibration_time", 100.0),
            production_time=data.get("production_time", 1000.0),
            force_field=data.get("force_field", "OPLS"),
            custom_settings=data.get("custom_settings", {})
        )
    
    def __repr__(self) -> str:
        """String representation of the parameters."""
        return (f"SimulationParameters(ensemble={self.ensemble}, "
                f"T={self.temperature}K, P={self.pressure}atm, "
                f"dt={self.timestep}fs, run={self.get_total_time()}ps)")
