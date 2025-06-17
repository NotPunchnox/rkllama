import os
import re
import logging
import requests
from pathlib import Path
import config

# Configure logger
logger = logging.getLogger("rkllama.model_utils")

# Global dictionaries for model mappings
SIMPLE_TO_FULL_MAP = {}  # Maps simplified names (e.g., "qwen2:3b") to full paths
FULL_TO_SIMPLE_MAP = {}  # Maps full paths to simplified names
TAG_TO_MODEL_MAP = {}  # Maps tag names from /api/tags to real model names

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

def get_huggingface_model_info(model_path):
    """
    Fetch model metadata from Hugging Face API if available.
    
    Args:
        model_path: HuggingFace repository path (e.g., 'c01zaut/Qwen2.5-3B-Instruct-RK3588-1.1.4')
        
    Returns:
        Dictionary with enhanced model metadata or None if not available
    """
    try:
        if not model_path or '/' not in model_path:
            return None
        
        # Get DEBUG_MODE from configuration
        debug_mode = config.is_debug_mode()
        
        # Extract repo_id from HUGGINGFACE_PATH
        url = f"https://huggingface.co/api/models/{model_path}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            # Process and enhance the metadata
            if 'tags' not in data:
                data['tags'] = []
            
            # Extract additional info from readme if available
            if 'cardData' not in data:
                data['cardData'] = {}
            
            # Try to extract parameter size from model name if not in cardData
            if 'params' not in data['cardData']:
                # Look for patterns like "7b", "3B", "1.5B" in model name or description
                param_pattern = re.search(r'(\d+\.?\d*)([bB])', model_path + ' ' + (data.get('description') or ''))
                if param_pattern:
                    size_value = float(param_pattern.group(1))
                    size_unit = param_pattern.group(2).lower()
                    # Convert to billions if needed
                    if size_unit == 'b':
                        data['cardData']['params'] = int(size_value * 1_000_000_000)
            
            # Extract important information from the description
            description = data.get('description', '')
            if description:
                # Look for model details in the description
                quant_pattern = re.search(r'([qQ]\d+_\d+|int4|int8|fp16|4bit|8bit)', description)
                if quant_pattern:
                    data['quantization'] = quant_pattern.group(1)
                
                # Check for mentions of specific architectures
                architectures = {
                    'llama': 'llama',
                    'mistral': 'mistral',
                    'qwen': 'qwen',
                    'deepseek': 'deepseek',
                    'phi': 'phi',
                    'gemma': 'gemma',
                    'baichuan': 'baichuan',
                    'yi': 'yi'
                }
                
                for arch_name, arch_value in architectures.items():
                    if arch_name.lower() in description.lower():
                        data['architecture'] = arch_value
                        if arch_name.lower() not in data['tags']:
                            data['tags'].append(arch_name.lower())
            
            # Try to extract language information
            languages = []
            language_patterns = {
                'english': 'en',
                'chinese': 'zh',
                'multilingual': None,  # Special case
                'french': 'fr',
                'german': 'de',
                'spanish': 'es',
                'japanese': 'ja'
            }
            
            for lang_name, lang_code in language_patterns.items():
                if lang_name.lower() in description.lower() or lang_name.lower() in ' '.join(data['tags']).lower():
                    if lang_name == 'multilingual':
                        # For multilingual models, add common languages
                        languages.extend(['en', 'zh', 'fr', 'de', 'es', 'ja'])
                    elif lang_code and lang_code not in languages:
                        languages.append(lang_code)
            
            # If we found languages, add them
            if languages:
                data['languages'] = list(set(languages))  # Remove duplicates
            elif 'en' not in data.get('languages', []):
                # Default to English if no languages detected
                data['languages'] = ['en']
            
            # Add RK tags if they exist
            rk_patterns = ['rk3588', 'rk3576', 'rkllm', 'rockchip']
            for pattern in rk_patterns:
                if pattern in model_path.lower() or pattern in ' '.join(data['tags']).lower() or pattern in description.lower():
                    if 'rockchip' not in data['tags']:
                        data['tags'].append('rockchip')
                    if pattern not in data['tags'] and pattern != 'rockchip':
                        data['tags'].append(pattern)
            
            # Add metadata about model capabilities
            if 'sibling_models' in data:
                for sibling in data.get('sibling_models', []):
                    if sibling.get('rfilename', '').endswith('.rkllm'):
                        data['has_rkllm'] = True
                        break
            
            # Extract license information
            if 'license' in data and data['license']:
                # Map HF license IDs to human-readable names
                license_mapping = {
                    'apache-2.0': 'Apache 2.0',
                    'mit': 'MIT',
                    'cc-by-4.0': 'Creative Commons Attribution 4.0',
                    'cc-by-sa-4.0': 'Creative Commons Attribution-ShareAlike 4.0',
                    'cc-by-nc-4.0': 'Creative Commons Attribution-NonCommercial 4.0',
                    'cc-by-nc-sa-4.0': 'Creative Commons Attribution-NonCommercial-ShareAlike 4.0'
                }
                
                license_id = data['license'].lower()
                data['license_name'] = license_mapping.get(license_id, data['license'])
                data['license_url'] = f"https://huggingface.co/{model_path}/blob/main/LICENSE"
            
            if debug_mode:
                logger.debug(f"Enhanced model info from HF API: {model_path}")
            
            return data
        else:
            if debug_mode:
                logger.debug(f"Failed to get HF data: {response.status_code}")
            return None
    except Exception as e:
        debug_mode = config.is_debug_mode()
        if debug_mode:
            logger.exception(f"Error fetching HF model info: {str(e)}")
        return None

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

def get_simplified_model_name(full_name, check_collision_map=True):
    """
    Convert a full model name to a simplified Ollama-style name
    
    Args:
        full_name: The full model name/path
        check_collision_map: If True, check if there's already a collision-aware name
        
    Returns:
        A simplified name like "qwen2.5-coder:7b"
    """
    # Handle paths - extract just the directory name
    if os.path.sep in full_name:
        full_name = os.path.basename(os.path.normpath(full_name))
        
    # First check if we already have a collision-resolved name for this model
    if check_collision_map and full_name in FULL_TO_SIMPLE_MAP:
        return FULL_TO_SIMPLE_MAP[full_name]
    
    # Remove any file extension
    full_name = os.path.splitext(full_name)[0]
    
    # Extract model family
    model_family = ""
    model_variants = []
    
    # First, check for model variants throughout the name
    # We'll do this first to ensure we capture all variants regardless of position
    variant_patterns = [
        ('coder', r'(?i)(^|[-_\s])coder($|[-_\s])'),
        ('math', r'(?i)(^|[-_\s])math($|[-_\s])'),
        ('chat', r'(?i)(^|[-_\s])chat($|[-_\s])'),
        ('instruct', r'(?i)(^|[-_\s])instruct($|[-_\s])'),
        ('vision', r'(?i)(^|[-_\s])vision($|[-_\s])'),
        ('mini', r'(?i)(^|[-_\s])mini($|[-_\s])'),
        ('small', r'(?i)(^|[-_\s])small($|[-_\s])'),
        ('medium', r'(?i)(^|[-_\s])medium($|[-_\s])'),
        ('large', r'(?i)(^|[-_\s])large($|[-_\s])'),
    ]
    
    for variant_name, pattern in variant_patterns:
        if re.search(pattern, full_name) and variant_name not in model_variants:
            model_variants.append(variant_name)
    
    # Now handle model family identification
    if re.search(r'(?i)deepseek', full_name):
        model_family = 'deepseek'
    elif re.search(r'(?i)qwen\d*', full_name):
        match = re.search(r'(?i)(qwen\d*)', full_name)
        if match:
            model_family = match.group(1).lower()
            if '2' in model_family:
                model_family = 'qwen2.5'
            else:
                model_family = 'qwen'
    elif re.search(r'(?i)mistral', full_name):
        model_family = 'mistral'
        if re.search(r'(?i)(^|[-_\s])nemo($|[-_\s])', full_name) and 'nemo' not in model_variants:
            model_variants.append('nemo')
    elif re.search(r'(?i)tinyllama', full_name):
        model_family = 'tinyllama'
    elif re.search(r'(?i)llama[-_]?3', full_name):
        model_family = 'llama3'
    elif re.search(r'(?i)llama[-_]?2', full_name):
        model_family = 'llama2'
    elif re.search(r'(?i)llama', full_name):
        model_family = 'llama'
    elif re.search(r'(?i)phi-3', full_name):
        model_family = 'phi3'
    elif re.search(r'(?i)phi-2', full_name):
        model_family = 'phi2'
    elif re.search(r'(?i)phi', full_name):
        model_family = 'phi'
    else:
        # Default to the first part of the name as family
        # Example: "Phi-2" becomes "phi"
        model_family = re.split(r'[-_\d]', full_name)[0].lower()
    
    # Extract parameter size
    param_size = ""
    # Try to find a pattern like "7B" or "3b"
    size_match = re.search(r'(?i)(\d+\.?\d*)B', full_name)
    if size_match:
        param_size = size_match.group(1).lower() + 'b'
    else:
        # Try other number patterns
        size_match = re.search(r'[-_](\d+)(?:[-_]|$)', full_name)
        if size_match:
            size = size_match.group(1)
            if len(size) <= 2:  # Likely a small number like 3, 7
                param_size = size + 'b'
    
    # Combine family, variant, and size with the new naming convention
    if model_family:
        # When multiple variants are present, join them with hyphens
        base_part = model_family
        if model_variants:
            variant_part = "-".join(model_variants)
            base_part = f"{model_family}-{variant_part}"
            
        if param_size:
            return f"{base_part}:{param_size}"
        else:
            return base_part
    else:
        # Fallback to a simplified version of the original name
        return re.sub(r'[^a-zA-Z0-9]', '-', full_name).lower()

def get_original_model_path(simplified_name):
    """
    Look up the original model directory name from a simplified name
    
    Args:
        simplified_name: A simplified model name like "qwen2:3b"
        
    Returns:
        The original model directory name or None if not found
    """
    if simplified_name in SIMPLE_TO_FULL_MAP:
        return SIMPLE_TO_FULL_MAP[simplified_name]
    return None

def initialize_model_mappings():
    """
    Initialize the model name mappings by scanning the models directory
    This should be called at server startup
    """
    global SIMPLE_TO_FULL_MAP, FULL_TO_SIMPLE_MAP, TAG_TO_MODEL_MAP
    SIMPLE_TO_FULL_MAP.clear()
    FULL_TO_SIMPLE_MAP.clear()
    TAG_TO_MODEL_MAP.clear()
    
    # Use config module to get models directory path
    models_dir = config.get_path("models")
    
    if not os.path.exists(models_dir):
        logger.warning(f"Models directory not found: {models_dir}")
        return
    
    # First pass: Create simplified names for all models
    model_names = {}  # Maps simple name to a list of full model names
    
    for model_dir in os.listdir(models_dir):
        full_path = os.path.join(models_dir, model_dir)
        
        if os.path.isdir(full_path) and any(f.endswith('.rkllm') for f in os.listdir(full_path)):
            simple_name = get_simplified_model_name(model_dir)

            # Build tag alias mapping for Open WebUI
            TAG_TO_MODEL_MAP[simple_name] = model_dir    
    
            if simple_name not in model_names:
                model_names[simple_name] = []
            model_names[simple_name].append(model_dir)
    
    # Second pass: Handle collisions by detecting differences
    for simple_name, full_names in model_names.items():
        # If only one model has this simple name, no collision to handle
        if len(full_names) == 1:
            SIMPLE_TO_FULL_MAP[simple_name] = full_names[0]
            FULL_TO_SIMPLE_MAP[full_names[0]] = simple_name
            SIMPLE_TO_FULL_MAP[full_names[0]] = full_names[0]  # Allow direct lookup too
            logger.debug(f"Mapped model: {full_names[0]} -> {simple_name}")
            continue
        
        # We have a collision - multiple models with the same simple name
        logger.warning(f"Simplified name collision: {simple_name} for models {', '.join(full_names)}")
        
        # For each colliding model, detect its distinctive features
        model_features = {}
        
        # Look for distinctive features in each model
        for model_dir in full_names:
            features = {}
            
            # Check for quantization type
            for pattern in ['w4a16', 'w4a16_g32', 'w4a16_g64', 'w4a16_g128', 
                           'w8a8', 'w8a8_g128', 'w8a8_g256', 'w8a8_g512']:
                if pattern in model_dir.lower():
                    features['quant'] = pattern.replace('_', '-')
                    break
            
            # Check for optimization level
            opt_match = re.search(r'opt-(\d+)', model_dir.lower())
            if opt_match:
                features['opt'] = f"opt{opt_match.group(1)}"
            
            # Check for hybrid ratio
            ratio_match = re.search(r'ratio-(\d+\.\d+|\d+)', model_dir.lower())
            if ratio_match:
                features['ratio'] = f"r{ratio_match.group(1)}"
            
            model_features[model_dir] = features
        
        # Find the most distinctive feature across models
        feature_counts = {'quant': {}, 'opt': {}, 'ratio': {}}
        
        for model_dir, features in model_features.items():
            for feature_type, value in features.items():
                if value not in feature_counts[feature_type]:
                    feature_counts[feature_type][value] = 0
                feature_counts[feature_type][value] += 1
        
        # Choose the feature type that has the most unique values
        best_feature_type = None
        max_unique_values = 0
        
        for feature_type, values in feature_counts.items():
            unique_count = len(values)
            if unique_count > max_unique_values:
                max_unique_values = unique_count
                best_feature_type = feature_type
        
        # If we found a good feature to differentiate, use it
        if best_feature_type and max_unique_values > 1:
            for model_dir in full_names:
                if best_feature_type in model_features[model_dir]:
                    feature_value = model_features[model_dir][best_feature_type]
                    
                    # If all models in the collision have this feature, we need to use it for all
                    # to differentiate them (otherwise we'll still have collisions)
                    if len(feature_counts[best_feature_type]) == len(full_names):
                        new_name = f"{simple_name}-{feature_value}"
                    else:
                        # Only add the feature to models that have it
                        new_name = f"{simple_name}-{feature_value}" if feature_value else simple_name
                    
                    SIMPLE_TO_FULL_MAP[new_name] = model_dir
                    FULL_TO_SIMPLE_MAP[model_dir] = new_name
                    SIMPLE_TO_FULL_MAP[model_dir] = model_dir  # Allow direct lookup
                    logger.info(f"Differentiated model: {model_dir} -> {new_name}")
                else:
                    # This model doesn't have the distinctive feature
                    SIMPLE_TO_FULL_MAP[simple_name] = model_dir
                    FULL_TO_SIMPLE_MAP[model_dir] = simple_name
                    SIMPLE_TO_FULL_MAP[model_dir] = model_dir  # Allow direct lookup
                    logger.info(f"Kept original name: {model_dir} -> {simple_name}")
        else:
            # If we couldn't find a good feature, use numeric suffixes
            SIMPLE_TO_FULL_MAP[simple_name] = full_names[0]
            FULL_TO_SIMPLE_MAP[full_names[0]] = simple_name
            SIMPLE_TO_FULL_MAP[full_names[0]] = full_names[0]  # Allow direct lookup
            
            for i, model_dir in enumerate(full_names[1:], 1):
                new_name = f"{simple_name}-{i}"
                SIMPLE_TO_FULL_MAP[new_name] = model_dir
                FULL_TO_SIMPLE_MAP[model_dir] = new_name
                SIMPLE_TO_FULL_MAP[model_dir] = model_dir  # Allow direct lookup
                logger.info(f"Added numeric suffix: {model_dir} -> {new_name}")

def find_model_by_name(name):
    """
    Find the actual model directory name from a simplified name or direct name
    
    Args:
        name: A model name (either simplified like "qwen2:3b" or full path)
    
    Returns:
        The full model directory name or None if not found
    """
    # Try direct lookup first - maybe it's already the full path
    if name in SIMPLE_TO_FULL_MAP:
        return SIMPLE_TO_FULL_MAP[name]

    # Try Open WebUI tag map
    if name in TAG_TO_MODEL_MAP:
        return TAG_TO_MODEL_MAP[name]

    
    # Check if it's a fully qualified path that exists directly
    models_dir = config.get_path("models")
    direct_path = os.path.join(models_dir, name)
    if os.path.isdir(direct_path):
        # It exists directly, make sure we use the collision-aware name
        return name
    
    # Try case-insensitive matching
    for full_name in FULL_TO_SIMPLE_MAP.keys():
        if name.lower() == full_name.lower():
            return full_name
            
    # Check if any of the model directories contain the name
    for model_dir in os.listdir(models_dir):
        full_path = os.path.join(models_dir, model_dir)
        if os.path.isdir(full_path) and name.lower() in model_dir.lower():
            return model_dir
    
    # If we get here, the model was not found
    logger.error(f"Model not found: {name}")
    return None

def ensure_model_loaded(model_name):
    """
    Ensure a model is properly resolved to a valid directory path
    
    Args:
        model_name: A model name (either simplified or full)
    
    Returns:
        The resolved model directory name or None if not found
    """
    # Try looking up the model by name
    full_model_name = find_model_by_name(model_name)
    if not full_model_name:
        # As a last resort, try direct path
        models_dir = config.get_path("models")
        if os.path.exists(os.path.join(models_dir, model_name)):
            return model_name
        
        # Try case-insensitive directory matching
        for dir_name in os.listdir(models_dir):
            if os.path.isdir(os.path.join(models_dir, dir_name)) and model_name.lower() == dir_name.lower():
                return dir_name
        
        # If we reach here, the model truly wasn't found
        return None
    
    return full_model_name

import os
import re
from typing import Union

MODEL_SPECS = {
    "qwen2":    (4096, [r'(?i)qwen']),
    "mistral":  (4096,  [r'(?i)mistral']),
    "llama3":   (4096,  [r'(?i)llama[-_]?3']),
    "llama2":   (4096,  [r'(?i)llama[-_]?2']),
    "gemma":    (4096,  [r'(?i)gemma']),
    "phi":      (2048,  [r'(?i)phi']),
    "llama":    (4096,  [])  # fallback
}

def detect_family(text: str) -> str:
    return next((name for name, (_, patterns) in MODEL_SPECS.items()
                 for p in patterns if re.search(p, text)), "llama")

def get_context_length(model_name: str, models_path: str = "models") -> Union[int, str]:
    modelfile = os.path.join(models_path, model_name, "Modelfile")
    try:
        with open(modelfile, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        text = model_name
    family = detect_family(text)
    return MODEL_SPECS[family][0]

def get_property_modelfile(model_name: str, property: str, models_path: str = "models"):
    modelfile = os.path.join(models_path, model_name, "Modelfile")

    # Initialize an empty dictionary to store key-value pairs
    modelfile_dict = {}

    # Open and read the file
    try:
        with open(modelfile, 'r') as file:
            for line in file:
                line = line.strip()
                if '=' in line:
                    # Split the line into key and value (split on first '=')
                    key, value = line.split('=', 1)
                    modelfile_dict[key] = value
    except FileNotFoundError:
        logger.error(f"Error: File '{modelfile}' not found.")

    # Retrieve the value of the property
    return modelfile_dict.get(property, None)

