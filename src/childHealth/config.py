from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, ValidationError


class ProjectConfig(BaseModel):
    catalog_name: str
    schema_name: str
    random_forest_parameters: Dict[str, Any]  # Dictionary to hold Random Forest parameters
    lgb_parameters: Dict[str, Any]  # Dictionary to hold LightGBM parameters
    num_features: List[str]
    cat_features: List[str]
    target: str

    @classmethod
    def from_yaml(cls, config_path: str) -> "ProjectConfig":
        """Load configuration from a YAML file."""
        try:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
            return cls(**config_dict)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
        except ValidationError as e:
            raise ValueError(f"Validation error: {e}")


# Example usage:
# config = ProjectConfig.from_yaml('path/to/project_config.yml')
