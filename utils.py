"""
Utility functions for configuration management.

This module provides utilities for loading and processing configuration files,
including environment variable substitution.
"""

import os
from string import Template

import yaml


def load_config(config_file):
    """
    Load and process configuration file with environment variable substitution.

    Args:
        config_file (str): Path to YAML configuration file

    Returns:
        dict: Parsed configuration with environment variables substituted

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        KeyError: If required environment variable is missing
    """
    try:
        # Read the config file as a template
        with open(config_file) as f:
            template = Template(f.read())

        # Get environment variables
        env_vars = os.environ

        # Substitute environment variables in the template
        try:
            config_str = template.substitute(env_vars)
        except KeyError as e:
            missing_var = str(e).strip("'")
            raise KeyError(
                f"Environment variable '{missing_var}' required by config file is not set"
            )

        # Parse the YAML with substituted values
        try:
            config = yaml.safe_load(config_str)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML configuration: {str(e)}")

        return config

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
