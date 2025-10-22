"""
File Utilities

Helper functions for file I/O operations.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def load_yaml(filepath: str) -> Dict[str, Any]:
    """
    Load YAML file

    Args:
        filepath: Path to YAML file

    Returns:
        Dictionary with YAML contents

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        logger.debug(f"Loaded YAML from {filepath}")
        return data
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in {filepath}: {e}")
        raise


def save_yaml(data: Dict[str, Any], filepath: str, indent: int = 2):
    """
    Save dictionary to YAML file

    Args:
        data: Dictionary to save
        filepath: Path to save YAML file
        indent: Indentation level
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=indent, sort_keys=False)
        logger.debug(f"Saved YAML to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save YAML to {filepath}: {e}")
        raise


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary with JSON contents

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.debug(f"Loaded JSON from {filepath}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filepath}: {e}")
        raise


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2):
    """
    Save dictionary to JSON file

    Args:
        data: Dictionary to save
        filepath: Path to save JSON file
        indent: Indentation level
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)
        logger.debug(f"Saved JSON to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")
        raise


def ensure_dir(dirpath: str):
    """
    Ensure directory exists, create if it doesn't

    Args:
        dirpath: Path to directory
    """
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {dirpath}")


# Alias for consistency with different naming conventions
def ensure_directory(dirpath: str):
    """
    Alias for ensure_dir - ensures directory exists, creates if it doesn't

    Args:
        dirpath: Path to directory
    """
    ensure_dir(dirpath)


def list_files(directory: str, pattern: str = "*") -> List[Path]:
    """
    List files in directory matching pattern

    Args:
        directory: Path to directory
        pattern: Glob pattern (default: all files)

    Returns:
        List of Path objects
    """
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return []

    files = list(directory.glob(pattern))
    logger.debug(f"Found {len(files)} files matching '{pattern}' in {directory}")
    return files


def list_patient_profiles(profiles_dir: str = "profiles") -> List[Path]:
    """
    List all patient profile YAML files

    Args:
        profiles_dir: Path to profiles directory

    Returns:
        List of patient profile file paths
    """
    return list_files(profiles_dir, "patient_*.yaml")


def list_disease_models(diseases_dir: str = "diseases") -> List[Path]:
    """
    List all disease model YAML files

    Args:
        diseases_dir: Path to diseases directory

    Returns:
        List of disease model file paths
    """
    return list_files(diseases_dir, "*.yaml")


def get_patient_id_from_filename(filepath: Path) -> Optional[str]:
    """
    Extract patient ID from profile filename

    Args:
        filepath: Path to patient profile file

    Returns:
        Patient ID or None if not found

    Example:
        patient_001_alex_chen.yaml -> patient_001
    """
    filename = filepath.stem  # Get filename without extension
    parts = filename.split('_')
    if len(parts) >= 2 and parts[0] == 'patient':
        return f"{parts[0]}_{parts[1]}"
    return None


def get_disease_id_from_filename(filepath: Path) -> Optional[str]:
    """
    Extract disease ID from disease model filename

    Args:
        filepath: Path to disease model file

    Returns:
        Disease ID (filename without extension)

    Example:
        viral_uri.yaml -> viral_uri
    """
    return filepath.stem


def validate_yaml_schema(data: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Validate YAML data has required fields

    Args:
        data: Dictionary to validate
        required_fields: List of required field names

    Returns:
        True if valid

    Raises:
        ValueError: If required field is missing
    """
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    return True


def create_conversation_filepath(patient_id: str, day: int, conversations_dir: str = "data/conversations") -> Path:
    """
    Create filepath for conversation JSON

    Args:
        patient_id: Patient identifier
        day: Simulation day
        conversations_dir: Base conversations directory

    Returns:
        Path to conversation file

    Example:
        patient_001, day 5 -> data/conversations/patient_001/day_5_20241022_1423.json
    """
    from datetime import datetime

    conversations_dir = Path(conversations_dir)
    patient_dir = conversations_dir / patient_id
    ensure_dir(patient_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"day_{day}_{timestamp}.json"

    return patient_dir / filename


def create_state_filepath(patient_id: str, state_dir: str = "data/state") -> Path:
    """
    Create filepath for patient state JSON

    Args:
        patient_id: Patient identifier
        state_dir: Base state directory

    Returns:
        Path to state file

    Example:
        patient_001 -> data/state/patient_001_state.json
    """
    state_dir = Path(state_dir)
    ensure_dir(state_dir)

    filename = f"{patient_id}_state.json"
    return state_dir / filename


def file_exists(filepath: str) -> bool:
    """
    Check if file exists

    Args:
        filepath: Path to file

    Returns:
        True if file exists
    """
    return Path(filepath).exists()


def get_file_size(filepath: str) -> int:
    """
    Get file size in bytes

    Args:
        filepath: Path to file

    Returns:
        File size in bytes
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return 0
    return filepath.stat().st_size


def read_text_file(filepath: str) -> str:
    """
    Read text file contents

    Args:
        filepath: Path to text file

    Returns:
        File contents as string
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r') as f:
        return f.read()


def write_text_file(content: str, filepath: str):
    """
    Write text to file

    Args:
        content: Text content to write
        filepath: Path to file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        f.write(content)
    logger.debug(f"Wrote text to {filepath}")
