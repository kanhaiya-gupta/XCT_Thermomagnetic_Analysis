"""
Reproducibility Framework

Ensure scientific reproducibility through:
- Configuration management (YAML/JSON)
- Version tracking
- Seed management
- Provenance logging
- Analysis package export
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import hashlib
import logging
import numpy as np
import sys
import platform

logger = logging.getLogger(__name__)


class AnalysisConfig:
    """Configuration manager for analysis parameters."""

    def __init__(
        self,
        config_dict: Optional[Union[Dict[str, Any], "AnalysisConfig"]] = None,
        **kwargs,
    ):
        """
        Initialize configuration.

        Args:
            config_dict: Optional dictionary with configuration, or another AnalysisConfig instance
            **kwargs: Configuration parameters as keyword arguments
        """
        if config_dict is None:
            config_dict = {}
        elif isinstance(config_dict, AnalysisConfig):
            # If AnalysisConfig passed, extract its config dict
            config_dict = config_dict.config.copy()
        elif not isinstance(config_dict, dict):
            # Convert to dict if possible
            config_dict = dict(config_dict) if hasattr(config_dict, "__iter__") else {}

        # Merge kwargs into config_dict
        if isinstance(config_dict, dict):
            config_dict.update(kwargs)
        self.config = config_dict if isinstance(config_dict, dict) else {}
        self.timestamp = datetime.now().isoformat()

        # Allow direct attribute access for common parameters
        for key, value in self.config.items():
            setattr(self, key, value)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = self.config.copy()
        result["timestamp"] = self.timestamp
        result["version"] = self.get("version", "1.0.0")
        return result

    def save(self, output_path: Union[str, Path], format: str = "yaml") -> None:
        """
        Save configuration to file.

        Args:
            output_path: Path to save configuration
            format: 'yaml' or 'json'
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()

        if format.lower() == "yaml":
            with open(output_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        else:  # JSON
            with open(output_path, "w") as f:
                json.dump(config_dict, f, indent=2)

        logger.info(f"Configuration saved to {output_path}")

    @classmethod
    def load(cls, config_path: Union[str, Path]) -> "AnalysisConfig":
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            AnalysisConfig instance
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if config_path.suffix.lower() in [".yaml", ".yml"]:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f) or {}
        else:  # JSON
            with open(config_path, "r") as f:
                content = f.read().strip()
                if not content:
                    raise ValueError(f"Configuration file is empty: {config_path}")
                config_dict = json.loads(content)

        if config_dict is None:
            config_dict = {}

        if "config" in config_dict:
            config_data = config_dict["config"]
        else:
            # Remove metadata keys that shouldn't be in config
            metadata_keys = {"timestamp", "version"}
            config_data = {
                k: v for k, v in config_dict.items() if k not in metadata_keys
            }

        # Convert lists back to tuples for known tuple fields (JSON doesn't preserve tuples)
        tuple_fields = {"voxel_size", "spacing", "origin", "dimensions", "size"}
        for key, value in config_data.items():
            if key in tuple_fields and isinstance(value, list) and len(value) > 0:
                # Check if all elements are numbers
                if all(isinstance(x, (int, float)) for x in value):
                    config_data[key] = tuple(value)

        config = cls(config_data)

        return config


class ProvenanceTracker:
    """Track analysis provenance for reproducibility."""

    def __init__(self):
        """Initialize provenance tracker."""
        self.steps = []
        self.environment = self._capture_environment()
        self.start_time = datetime.now()

    def _capture_environment(self) -> Dict[str, Any]:
        """Capture computational environment."""
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "timestamp": datetime.now().isoformat(),
        }

    def add_step(
        self,
        step_name: str,
        parameters: Dict[str, Any],
        function_name: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add analysis step to provenance.

        Args:
            step_name: Name of the step
            function_name: Name of function called
            parameters: Parameters used
            inputs: Optional input data hashes
            outputs: Optional output data hashes
        """
        step = {
            "step_name": step_name,
            "step": step_name,  # Alias for compatibility
            "function_name": function_name or step_name,
            "function": function_name or step_name,  # Alias
            "parameters": parameters,
            "timestamp": datetime.now().isoformat(),
            "inputs": inputs or {},
            "outputs": outputs or {},
        }
        self.steps.append(step)
        logger.debug(f"Provenance step added: {step_name}")

    def compute_data_hash(self, data: Any) -> str:
        """
        Compute hash of data for tracking.

        Args:
            data: Data to hash

        Returns:
            Hash string
        """
        if isinstance(data, np.ndarray):
            # Hash array
            return hashlib.md5(data.tobytes()).hexdigest()
        elif isinstance(data, (dict, list)):
            # Hash JSON representation
            return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
        else:
            # Hash string representation
            return hashlib.md5(str(data).encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert provenance to dictionary."""
        return {
            "environment": self.environment,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "n_steps": len(self.steps),
            "steps": self.steps,
        }

    def save(self, output_path: Union[str, Path]) -> None:
        """
        Save provenance to file.

        Args:
            output_path: Path to save provenance
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        provenance_dict = self.to_dict()

        with open(output_path, "w") as f:
            json.dump(provenance_dict, f, indent=2)

        logger.info(f"Provenance saved to {output_path}")

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get analysis history (list of steps).

        Returns:
            List of step dictionaries
        """
        return self.steps

    def export(self) -> Dict[str, Any]:
        """
        Export provenance as dictionary.

        Returns:
            Dictionary with provenance information
        """
        return {
            "history": self.steps,
            "timestamp": datetime.now().isoformat(),
            "environment": self.environment,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "n_steps": len(self.steps),
        }


class SeedManager:
    """Manage random seeds for reproducibility."""

    def __init__(self, base_seed: Optional[int] = None, seed: Optional[int] = None):
        """
        Initialize seed manager.

        Args:
            base_seed: Base seed value (if None, uses timestamp)
            seed: Alias for base_seed (for compatibility)
        """
        # Use seed if provided, otherwise base_seed, otherwise timestamp
        if seed is not None:
            base_seed = seed
        elif base_seed is None:
            base_seed = int(datetime.now().timestamp() * 1000) % (2**31)

        self.base_seed = base_seed
        self.seed_counter = 0
        self.used_seeds = []

    def get_seed(self, name: Optional[str] = None) -> int:
        """
        Get next seed value.

        Args:
            name: Optional name for seed

        Returns:
            Seed value
        """
        seed = self.base_seed + self.seed_counter
        self.seed_counter += 1
        self.used_seeds.append(
            {
                "seed": seed,
                "name": name or f"seed_{self.seed_counter}",
                "counter": self.seed_counter,
            }
        )
        return seed

    def set_seed(self, seed: Optional[int] = None) -> None:
        """
        Set seed for random number generators.

        Args:
            seed: Seed value (if None, uses base_seed)
        """
        if seed is None:
            seed = self.base_seed
        np.random.seed(seed)
        self.used_seeds.append(
            {"seed": seed, "name": "manual", "counter": len(self.used_seeds)}
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_seed": self.base_seed,
            "n_seeds_used": len(self.used_seeds),
            "used_seeds": self.used_seeds,
        }


def track_analysis_provenance(
    analysis_steps: List[Dict[str, Any]], output_path: Optional[Union[str, Path]] = None
) -> ProvenanceTracker:
    """
    Track analysis provenance from list of steps.

    Args:
        analysis_steps: List of step dictionaries
        output_path: Optional path to save provenance

    Returns:
        ProvenanceTracker instance
    """
    tracker = ProvenanceTracker()

    for step in analysis_steps:
        tracker.add_step(
            step_name=step.get("step_name", "unknown"),
            function_name=step.get("function_name", "unknown"),
            parameters=step.get("parameters", {}),
            inputs=step.get("inputs"),
            outputs=step.get("outputs"),
        )

    if output_path:
        tracker.save(output_path)

    return tracker


def export_reproducibility_package(
    output_dir: Union[str, Path],
    config: Optional[AnalysisConfig] = None,
    provenance: Optional[ProvenanceTracker] = None,
    seed_manager: Optional[SeedManager] = None,
    data_files: Optional[List[Union[str, Path]]] = None,
    code_files: Optional[List[Union[str, Path]]] = None,
    results: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Export complete reproducibility package.

    Args:
        output_dir: Output directory
        config: Optional analysis configuration
        provenance: Optional provenance tracker
        seed_manager: Optional seed manager
        data_files: Optional list of data files to include
        code_files: Optional list of code files to include
        results: Optional analysis results

    Returns:
        Path to exported package
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    if config:
        config.save(output_dir / "analysis_config.yaml")

    # Save provenance
    if provenance:
        provenance.save(output_dir / "provenance.json")

    # Save seed information
    if seed_manager:
        with open(output_dir / "seeds.json", "w") as f:
            json.dump(seed_manager.to_dict(), f, indent=2)

    # Copy data files
    if data_files:
        data_dir = output_dir / "data"
        data_dir.mkdir(exist_ok=True)
        for data_file in data_files:
            src = Path(data_file)
            if src.exists():
                import shutil

                shutil.copy2(src, data_dir / src.name)

    # Copy code files
    if code_files:
        code_dir = output_dir / "code"
        code_dir.mkdir(exist_ok=True)
        for code_file in code_files:
            src = Path(code_file)
            if src.exists():
                import shutil

                shutil.copy2(src, code_dir / src.name)

    # Save results
    if results:
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Create README
    env_info = provenance.environment if provenance else "Not available"
    readme_content = f"""# Reproducibility Package

Generated: {datetime.now().isoformat()}

## Contents

- `analysis_config.yaml`: Analysis configuration
- `provenance.json`: Analysis provenance
- `seeds.json`: Random seed information
- `results.json`: Analysis results
- `data/`: Input data files
- `code/`: Code files used

## Reproducing Analysis

1. Load configuration: `config = AnalysisConfig.load('analysis_config.yaml')`
2. Set seeds: Use seeds from `seeds.json`
3. Run analysis with same parameters
4. Compare results with `results.json`

## Environment

{env_info}
"""

    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)

    logger.info(f"Reproducibility package exported to {output_dir}")
    return output_dir


def save_analysis_config(
    config: Union[Dict[str, Any], AnalysisConfig],
    output_path: Union[str, Path],
    format: Optional[str] = None,
) -> None:
    """
    Save analysis configuration.

    Args:
        config: Configuration dictionary or AnalysisConfig instance
        output_path: Path to save
        format: 'yaml' or 'json' (if None, detected from file extension)
    """
    output_path = Path(output_path)
    if format is None:
        # Detect format from file extension
        if output_path.suffix.lower() in [".yaml", ".yml"]:
            format = "yaml"
        else:
            format = "json"

    if isinstance(config, AnalysisConfig):
        analysis_config = config
    else:
        analysis_config = AnalysisConfig(config)
    analysis_config.save(output_path, format)


def load_analysis_config(config_path: Union[str, Path]) -> AnalysisConfig:
    """
    Load analysis configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        AnalysisConfig instance
    """
    return AnalysisConfig.load(config_path)
