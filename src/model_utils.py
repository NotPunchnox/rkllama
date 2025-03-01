import os
import re
import logging

# Configure logger
logger = logging.getLogger("rkllama.model_utils")

# Dictionary to store mappings between simplified names and actual model paths
model_name_mappings = {}

# Mapping from RKLLM quantization types to Ollama-style formats
QUANT_MAPPING = {
    'w4a16': 'Q4_0',
    'w4a16_g32': 'Q4_K_M',
    'w4a16_g64': 'Q4_K_M',
    'w4a16_g128': 'Q4_K_M',
    'w8a8': 'Q8_0',
    'w8a8_g128': 'Q8_K_M',
    'w8a8_g256': 'Q8_K_M',
    'w8a8_g512': 'Q8_K_M',
}

def extract_model_details(model_name):
    """
    Extract model parameter size and quantization type from model name
    
    Args:
        model_name: Model name or file path
        
    Returns:
        Dictionary with parameter_size and quantization_level
    """
    # Initialize default values
    details = {
        "parameter_size": "Unknown",
        "quantization_level": "Unknown"
    }
    
    # Remove path and extension if present
    if isinstance(model_name, str):
        basename = os.path.basename(model_name).replace('.rkllm', '')
    else:
        basename = str(model_name)
    
    # Extract parameter size (e.g., 3B, 7B, 13B)
    param_size_match = re.search(r'(\d+\.?\d*)B', basename)
    if param_size_match:
        size = param_size_match.group(1)
        # Convert to standard format (3B, 7B, 13B, etc)
        if '.' in size:
            # For sizes like 1.1B, 2.7B
            details["parameter_size"] = f"{size}B"
        else:
            # For sizes like 3B, 7B
            details["parameter_size"] = f"{size}B"
    
    # Extract quantization type
    # Look for common quantization patterns
    quant_patterns = [
        ('w4a16', r'w4a16(?!_g)'),
        ('w4a16_g32', r'w4a16_g32'),
        ('w4a16_g64', r'w4a16_g64'),
        ('w4a16_g128', r'w4a16_g128'),
        ('w8a8', r'w8a8(?!_g)'),
        ('w8a8_g128', r'w8a8_g128'),
        ('w8a8_g256', r'w8a8_g256'),
        ('w8a8_g512', r'w8a8_g512')
    ]
    
    # Mapping to Ollama-style quantization names
    quant_mapping = {
        'w4a16': 'Q4_0',
        'w4a16_g32': 'Q4_K_M',
        'w4a16_g64': 'Q4_K_M',
        'w4a16_g128': 'Q4_K_M',
        'w8a8': 'Q8_0',
        'w8a8_g128': 'Q8_K_M',
        'w8a8_g256': 'Q8_K_M',
        'w8a8_g512': 'Q8_K_M'
    }
    
    for quant_type, pattern in quant_patterns:
        if re.search(pattern, basename, re.IGNORECASE):
            # Use Ollama-style quantization name if available
            details["quantization_level"] = quant_mapping.get(quant_type, quant_type)
            break
            
    return details

def get_simplified_model_name(model_path):
    """
    Convert RKLLM model names to Ollama-style simplified names
    
    Args:
        model_path: Path to the model directory or filename
        
    Returns:
        A simplified name like 'qwen2.5:3b' instead of 'Qwen2.5-3B-Instruct-rk3588-w8a8-opt-0-hybrid-ratio-1.0'
    """
    # Extract just the directory/file name without path
    model_name = os.path.basename(model_path).replace('.rkllm', '')
    
    # Store original name for logging
    original_name = model_name
    
    # Match common model families and sizes
    simplified_name = None
    
    # Check for Qwen models
    qwen_match = re.search(r'(?i)Qwen[-_]?(\d+\.?\d*)[-_]?(\d+)B', model_name)
    if qwen_match:
        version = qwen_match.group(1)
        size = qwen_match.group(2).lower()
        simplified_name = f"qwen{version}:{size}b"
    
    # Check for TinyLlama models
    tiny_match = re.search(r'(?i)TinyLlama[-_]?(\d+\.?\d*)B', model_name)
    if tiny_match:
        size = tiny_match.group(1).lower()
        simplified_name = f"tinyllama:{size}b"
    
    # Check for Llama models
    llama_match = re.search(r'(?i)Llama[-_]?(\d+)[-_]?(\d+)B', model_name)
    if llama_match:
        version = llama_match.group(1)
        size = llama_match.group(2).lower()
        simplified_name = f"llama{version}:{size}b"
    
    # Check for Mistral models
    mistral_match = re.search(r'(?i)Mistral[-_]?(\d+)B', model_name)
    if mistral_match:
        size = mistral_match.group(1).lower()
        simplified_name = f"mistral:{size}b"
    
    # Check for DeepSeek models
    deepseek_match = re.search(r'(?i)DeepSeek[-_]?(\d+)B', model_name)
    if deepseek_match:
        size = deepseek_match.group(1).lower()
        simplified_name = f"deepseek:{size}b"
    
    # If we couldn't match a known pattern, create a generic simplified name
    if not simplified_name:
        # Remove common suffixes and prefixes
        name = model_name
        name = re.sub(r'[-_]rk\d{4}.*$', '', name)  # Remove RK platform suffix
        name = re.sub(r'[-_]Instruct.*$', '', name)  # Remove Instruct suffix
        name = re.sub(r'[-_]Chat.*$', '', name)      # Remove Chat suffix
        name = re.sub(r'[-_]v\d+\.\d+.*$', '', name) # Remove version suffix
        
        # Try to find size marker
        size_match = re.search(r'[-_]?(\d+)B', name)
        if size_match:
            base_name = name[:size_match.start()].lower().replace('_', '').replace('-', '')
            size = size_match.group(1).lower()
            simplified_name = f"{base_name}:{size}b"
        else:
            # If no size found, just use lowercase name
            simplified_name = name.lower().replace('_', '').replace('-', '')
            simplified_name = f"{simplified_name}:latest"
    
    logger.debug(f"Converted model name '{original_name}' to '{simplified_name}'")
    
    # Store the mapping for reverse lookup
    model_name_mappings[simplified_name] = model_path
    
    return simplified_name

def get_original_model_path(simplified_name):
    """
    Look up the original model path from a simplified name
    
    Args:
        simplified_name: Simplified model name like 'qwen2.5:3b'
        
    Returns:
        The original model path, or None if not found
    """
    return model_name_mappings.get(simplified_name)

def initialize_model_mappings():
    """
    Scan the models directory and initialize mappings between simplified 
    and full model names to ensure they're available even without calling /api/tags
    """
    models_dir = os.path.expanduser("~/RKLLAMA/models")
    if not os.path.exists(models_dir):
        return
    
    for subdir in os.listdir(models_dir):
        subdir_path = os.path.join(models_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".rkllm"):
                    # Create the mapping for this model
                    simple_name = get_simplified_model_name(subdir)
                    logger.debug(f"Initialized mapping: {simple_name} -> {subdir}")
                    break

def find_model_by_name(model_name):
    """
    Find a model by name, handling both simplified and full names.
    
    Args:
        model_name: Either a simplified name like 'qwen:3b' or a full name
        
    Returns:
        The full model name/path if found, or None if not found
    """
    # First check if this is a simplified name we already know
    original_path = get_original_model_path(model_name)
    if original_path:
        return original_path
    
    # If not found in mappings, check if the model directory exists directly
    model_dir = os.path.expanduser(f"~/RKLLAMA/models/{model_name}")
    if os.path.exists(model_dir):
        return model_name
    
    # If still not found, try to match by pattern (reverse lookup)
    # This is more expensive but helps with compatibility
    models_dir = os.path.expanduser("~/RKLLAMA/models")
    if os.path.exists(models_dir):
        for subdir in os.listdir(models_dir):
            subdir_path = os.path.join(models_dir, subdir)
            if os.path.isdir(subdir_path):
                # Check if any files with .rkllm extension exist
                has_rkllm = any(f.endswith(".rkllm") for f in os.listdir(subdir_path))
                if has_rkllm:
                    # Get the simplified name for this model
                    simple_name = get_simplified_model_name(subdir)
                    # Store in mappings for future use
                    model_name_mappings[simple_name] = subdir
                    
                    # Check if this matches what we're looking for
                    if model_name.lower() == simple_name.lower():
                        return subdir
                    
                    # Also check without version numbers (e.g., 'qwen' matches 'qwen:3b')
                    base_simple = simple_name.split(':')[0]
                    base_input = model_name.split(':')[0]
                    if base_input.lower() == base_simple.lower():
                        return subdir
    
    # Not found
    return None
