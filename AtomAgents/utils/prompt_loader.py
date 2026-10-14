"""
Prompt Loader implementation.

This utility loads and manages prompt templates from YAML files.
"""
import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path

from langchain_core.prompts import PromptTemplate

class PromptLoader:
    """
    PromptLoader is responsible for loading and managing prompt templates from YAML files.
    
    This class handles:
    1. Loading prompt templates from YAML files in a specified directory
    2. Providing access to specific prompts by name
    3. Creating PromptTemplate objects from the loaded templates
    """
    
    def __init__(
        self,
        templates_dir: Optional[str] = None,
        default_template_vars: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the PromptLoader.
        
        Args:
            templates_dir: Directory containing prompt YAML files
                           (defaults to project's templates/prompts directory)
            default_template_vars: Default variables to include in all prompts
        """
        # Determine templates directory
        if templates_dir is None:
            # Try to find the default templates directory
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent  # Go up to project root
            self.templates_dir = os.path.join(project_root, "templates", "prompts")
        else:
            self.templates_dir = templates_dir
            
        # Store default template variables
        self.default_template_vars = default_template_vars or {}
        
        # Initialize prompt templates storage
        self.templates: Dict[str, Dict[str, Any]] = {}
        self.prompt_templates: Dict[str, PromptTemplate] = {}
        
        # Load templates if the directory exists
        if os.path.isdir(self.templates_dir):
            self.load_templates()
        else:
            print(f"Warning: Templates directory not found at {self.templates_dir}")
    
    def load_templates(self):
        """
        Load all prompt templates from YAML files in the templates directory.
        """
        # Clear existing templates
        self.templates = {}
        
        # Find all YAML files
        yaml_files = []
        for file in os.listdir(self.templates_dir):
            if file.endswith(('.yaml', '.yml')):
                yaml_files.append(os.path.join(self.templates_dir, file))
        
        # Load each YAML file
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r') as f:
                    # Load the YAML content
                    template_data = yaml.safe_load(f)
                    
                    # Check if it's a dictionary of templates
                    if not isinstance(template_data, dict):
                        print(f"Warning: Invalid template format in {yaml_file}")
                        continue
                    
                    # Add templates to our collection
                    self.templates.update(template_data)
                    
            except Exception as e:
                print(f"Error loading template file {yaml_file}: {str(e)}")
    
    def get_template(self, template_name: str) -> Dict[str, Any]:
        """
        Get the raw template data by name.
        
        Args:
            template_name: Name of the template to retrieve
            
        Returns:
            Template data dictionary
            
        Raises:
            ValueError: If template not found
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        return self.templates[template_name]
    
    def get_prompt(self, template_name: str) -> PromptTemplate:
        """
        Get a PromptTemplate object for the specified template.
        
        Args:
            template_name: Name of the template to retrieve
            
        Returns:
            PromptTemplate instance
            
        Raises:
            ValueError: If template not found
        """
        # Check if we've already created this PromptTemplate
        if template_name in self.prompt_templates:
            return self.prompt_templates[template_name]
        
        # Get the template data
        template_data = self.get_template(template_name)
        
        # Ensure the template has the required fields
        if 'template' not in template_data:
            raise ValueError(f"Template '{template_name}' is missing 'template' field")
        
        # Get input variables, falling back to empty list
        input_variables = template_data.get('input_variables', [])
        
        # Get partial variables, combining defaults with template-specific ones
        partial_vars = self.default_template_vars.copy()
        if 'partial_variables' in template_data:
            partial_vars.update(template_data['partial_variables'])
            
        # Create the PromptTemplate
        # If the template has a 'partial_variables' field, use it
        # Otherwise, use the default template variables
        if 'partial_variables' in template_data:
            prompt = PromptTemplate(
                template=template_data['template'],
                input_variables=input_variables,
                partial_variables=partial_vars if partial_vars else None
            )
        else:
            prompt = PromptTemplate(
                template=template_data['template'],
                input_variables=input_variables,
            )
        
        # Cache the prompt template
        self.prompt_templates[template_name] = prompt
        
        return prompt
    
    def list_templates(self) -> List[str]:
        """
        Get a list of all available template names.
        
        Returns:
            List of template names
        """
        return list(self.templates.keys())
    
    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """
        Get metadata about a specific template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Dictionary with template metadata
            
        Raises:
            ValueError: If template not found
        """
        template_data = self.get_template(template_name)
        
        # Return relevant metadata
        return {
            "name": template_name,
            "description": template_data.get("description", "No description"),
            "input_variables": template_data.get("input_variables", []),
            "has_partial_variables": bool(template_data.get("partial_variables")),
            "example_input": template_data.get("example_input", {}),
        }
    
    def reload(self):
        """
        Reload all prompt templates from disk.
        """
        # Clear cached prompt templates
        self.prompt_templates = {}
        
        # Reload templates
        self.load_templates()
    
    def add_template(self, template_name: str, template_data: Dict[str, Any]):
        """
        Add or update a template in memory (not saved to disk).
        
        Args:
            template_name: Name of the template
            template_data: Template definition data
        """
        self.templates[template_name] = template_data
        
        # Remove cached prompt template if it exists
        if template_name in self.prompt_templates:
            del self.prompt_templates[template_name]
