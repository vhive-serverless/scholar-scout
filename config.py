import os
import yaml
from pathlib import Path
from string import Template

def load_config(config_path=None):
    """Load configuration from YAML file with environment variable substitution."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yml"
    
    with open(config_path) as f:
        # Read the file content
        template = Template(f.read())
        
        # Substitute environment variables
        yaml_content = template.safe_substitute(
            GMAIL_USERNAME=os.getenv('GMAIL_USERNAME', ''),
            GMAIL_PASSWORD=os.getenv('GMAIL_PASSWORD', ''),
            SLACK_API_TOKEN=os.getenv('SLACK_API_TOKEN', ''),
            PPLX_API_KEY=os.getenv('PPLX_API_KEY', '')
        )
        
        # Load YAML with substituted values
        config = yaml.safe_load(yaml_content)
        
        # Verify required credentials are present
        if not config['email']['username'] or not config['email']['password']:
            raise ValueError("Gmail credentials not found in environment variables")
            
        return config 