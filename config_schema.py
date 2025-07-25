from enum import Enum
from typing import Dict, Any, Optional, List, Union, Type, TypeVar, Generic, get_type_hints

class FieldType(Enum):
    """Enumeration of field types for configuration schema"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float" 
    BOOLEAN = "boolean"
    LIST = "list"
    PATH = "path"

T = TypeVar('T')

class ConfigField(Generic[T]):
    """Definition of a configuration field with type information and validation"""
    
    def __init__(
        self, 
        field_type: FieldType,
        default: T, 
        description: str = "",
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        options: Optional[List[Any]] = None,
        item_type: Optional[FieldType] = None,
        required: bool = False
    ):
        self.field_type = field_type
        self.default = default
        self.description = description
        self.min_value = min_value
        self.max_value = max_value
        self.options = options
        self.item_type = item_type
        self.required = required
        
    def validate(self, value: Any) -> T:
        """Validate a value against this field definition"""
        if value is None:
            if self.required:
                raise ValueError(f"Field is required but value is None")
            return self.default
            
        # Type conversion based on field_type
        converted_value = self._convert_value(value)
            
        # Range validation for numeric types
        if self.field_type in [FieldType.INTEGER, FieldType.FLOAT]:
            if self.min_value is not None and converted_value < self.min_value:
                raise ValueError(f"Value {converted_value} is less than minimum {self.min_value}")
            if self.max_value is not None and converted_value > self.max_value:
                raise ValueError(f"Value {converted_value} is greater than maximum {self.max_value}")
                
        # Options validation
        if self.options is not None and converted_value not in self.options:
            raise ValueError(f"Value {converted_value} is not in allowed options: {self.options}")
            
        return converted_value
        
    def _convert_value(self, value: Any) -> T:
        """Convert a value to the appropriate type based on field_type"""
        try:
            if self.field_type == FieldType.STRING:
                return str(value)
            elif self.field_type == FieldType.INTEGER:
                if isinstance(value, str):
                    return int(value)
                return int(value)
            elif self.field_type == FieldType.FLOAT:
                if isinstance(value, str):
                    return float(value)
                return float(value)
            elif self.field_type == FieldType.BOOLEAN:
                if isinstance(value, str):
                    return value.lower() in ('true', 'yes', '1', 'on', 'y')
                return bool(value)
            elif self.field_type == FieldType.LIST:
                if isinstance(value, str):
                    items = [item.strip() for item in value.split(',') if item.strip()]
                    if self.item_type:
                        # Convert each item to the specified type
                        temp_field = ConfigField(self.item_type, None)
                        return [temp_field._convert_value(item) for item in items]
                    return items
                elif isinstance(value, list):
                    return value
                else:
                    raise ValueError(f"Cannot convert {value} to list")
            elif self.field_type == FieldType.PATH:
                return str(value)
            else:
                return value
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to convert value {value} to {self.field_type.value}: {str(e)}")

class ConfigSectionSchema:
    """Schema definition for a configuration section"""
    
    def __init__(self, description: str = ""):
        self.description = description
        self.fields: Dict[str, ConfigField] = {}
        
    def add_field(self, name: str, field: ConfigField):
        """Add a field to this section schema"""
        self.fields[name] = field
        return self
        
    def string(self, name: str, default: str = "", description: str = "", options: List[str] = None, required: bool = False):
        """Add a string field to this section schema"""
        self.fields[name] = ConfigField(
            FieldType.STRING, 
            default, 
            description,
            options=options,
            required=required
        )
        return self
        
    def integer(self, name: str, default: int = 0, description: str = "", min_value: int = None, 
                max_value: int = None, required: bool = False):
        """Add an integer field to this section schema"""
        self.fields[name] = ConfigField(
            FieldType.INTEGER, 
            default, 
            description,
            min_value=min_value,
            max_value=max_value,
            required=required
        )
        return self
        
    def float(self, name: str, default: float = 0.0, description: str = "", 
              min_value: float = None, max_value: float = None, required: bool = False):
        """Add a float field to this section schema"""
        self.fields[name] = ConfigField(
            FieldType.FLOAT, 
            default, 
            description,
            min_value=min_value,
            max_value=max_value,
            required=required
        )
        return self
        
    def boolean(self, name: str, default: bool = False, description: str = "", required: bool = False):
        """Add a boolean field to this section schema"""
        self.fields[name] = ConfigField(
            FieldType.BOOLEAN, 
            default, 
            description,
            required=required
        )
        return self
        
    def list(self, name: str, default: List = None, description: str = "", 
             item_type: FieldType = None, required: bool = False):
        """Add a list field to this section schema"""
        if default is None:
            default = []
        self.fields[name] = ConfigField(
            FieldType.LIST, 
            default, 
            description,
            item_type=item_type,
            required=required
        )
        return self
    
    def path(self, name: str, default: str = "", description: str = "", required: bool = False):
        """Add a path field to this section schema"""
        self.fields[name] = ConfigField(
            FieldType.PATH,
            default,
            description,
            required=required
        )
        return self
        
    def validate_section(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate values against this section schema"""
        validated = {}
        
        # First, apply defaults for all fields
        for name, field in self.fields.items():
            validated[name] = field.default
            
        # Then override with provided values
        if values:
            for name, value in values.items():
                if name in self.fields:
                    try:
                        validated[name] = self.fields[name].validate(value)
                    except ValueError as e:
                        raise ValueError(f"Validation error for {name}: {str(e)}")
                else:
                    # Keep unknown fields, but don't validate them
                    validated[name] = value
                    
        return validated

class ConfigSchema:
    """Schema definition for the entire configuration"""
    
    def __init__(self):
        self.sections: Dict[str, ConfigSectionSchema] = {}
        
    def add_section(self, name: str, section: ConfigSectionSchema = None, description: str = ""):
        """Add a section to this schema"""
        if section is None:
            section = ConfigSectionSchema(description)
        self.sections[name] = section
        return section
        
    def get_section(self, name: str) -> ConfigSectionSchema:
        """Get a section from this schema"""
        return self.sections.get(name)
        
    def validate(self, config: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Validate a configuration against this schema"""
        validated = {}
        
        # First, apply defaults for all sections
        for section_name, section_schema in self.sections.items():
            validated[section_name] = section_schema.validate_section({})
            
        # Then override with provided values
        if config:
            for section_name, section_values in config.items():
                if section_name in self.sections:
                    validated[section_name] = self.sections[section_name].validate_section(section_values)
                else:
                    # Keep unknown sections, but don't validate them
                    validated[section_name] = section_values
                    
        return validated

def create_rkllama_schema() -> ConfigSchema:
    """Create and return the RKLLAMA configuration schema"""
    schema = ConfigSchema()
    
    # Server section
    server = schema.add_section("server", description="Server configuration settings")
    server.integer("port", 8080, "Server port number", min_value=1, max_value=65535)
    server.string("host", "0.0.0.0", "Server host address")
    server.boolean("debug", False, "Enable debug mode")
    
    # Paths section
    paths = schema.add_section("paths", description="Path configuration")
    paths.path("models", "models", "Path to model files")
    paths.path("logs", "logs", "Path to log files")
    paths.path("data", "data", "Path to data files")
    paths.path("src", "src", "Path to source files")
    paths.path("lib", "lib", "Path to library files")
    paths.path("temp", "temp", "Path to temporary files")
    
    # Model section
    model = schema.add_section("model", description="Model configuration")
    model.string("default", "", "Default model to use")
    model.string("default_temperature", 0.5, "Default temperature for the model to use")
    model.string("default_enable_thinking", False, "Default Enable Thinking for the model to use")
    model.string("default_num_ctx", 16384, "Default Context Length for the model to use")
    model.string("default_max_new_tokens", 16384, "Default Max New Tokens for the model to use")
    model.string("default_top_k", 7, "Default Top K for the model to use")
    model.string("default_top_p", 0.5, "Default Top P for the model to use")
    model.string("default_repeat_penalty", 1.1, "Default Repeat Penalty for the model to use")
    model.string("default_frequency_penalty", 0.0, "Default Frequency Penalty for the model to use")
    model.string("default_presence_penalty", 0.0, "Default Presence Penalty for the model to use")
    model.string("default_mirostat", 0, "Default Mirostat for the model to use")
    model.string("default_mirostat_tau", 3, "Default Mirostat Tau for the model to use")
    model.string("default_mirostat_eta", 0.1, "Default Mirostat Eta for the model to use")
    
    # Platform section
    platform = schema.add_section("platform", description="Platform configuration")
    platform.string("processor", "rk3588", "Target processor", 
                   options=["rk3588", "rk3576"])
    
    return schema

# Create the global schema instance
RKLLAMA_SCHEMA = create_rkllama_schema()
