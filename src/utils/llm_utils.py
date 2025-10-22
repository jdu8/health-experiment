"""
LLM Utilities

Helper functions for loading and managing LLM models.
"""

import logging
import torch
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Global cache for loaded models
_MODEL_CACHE: Dict[str, Any] = {}


def check_transformers_available() -> bool:
    """
    Check if transformers library is available

    Returns:
        True if transformers is installed
    """
    try:
        import transformers
        return True
    except ImportError:
        return False


def check_cuda_available() -> bool:
    """
    Check if CUDA/GPU is available

    Returns:
        True if CUDA is available
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device(prefer_cuda: bool = True) -> str:
    """
    Get appropriate device for model loading

    Args:
        prefer_cuda: Whether to prefer CUDA if available

    Returns:
        Device string ('cuda' or 'cpu')
    """
    if prefer_cuda and check_cuda_available():
        device = "cuda"
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("Using CPU device")

    return device


def load_model_and_tokenizer(
    model_name: str,
    device: str = "auto",
    quantization: Optional[str] = None,
    use_cache: bool = True
) -> tuple:
    """
    Load HuggingFace model and tokenizer with proper configuration

    Args:
        model_name: HuggingFace model name (e.g., "mistralai/Mistral-7B-Instruct-v0.2")
        device: Device to load on ('cuda', 'cpu', or 'auto')
        quantization: Quantization type ('4bit', '8bit', or None)
        use_cache: Whether to use cached models

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ImportError: If required libraries not installed
        RuntimeError: If model loading fails
    """
    # Check if transformers is available
    if not check_transformers_available():
        raise ImportError(
            "transformers library not installed. "
            "Install with: pip install transformers accelerate"
        )

    # Check cache
    cache_key = f"{model_name}_{device}_{quantization}"
    if use_cache and cache_key in _MODEL_CACHE:
        logger.info(f"Using cached model: {model_name}")
        return _MODEL_CACHE[cache_key]

    logger.info(f"Loading model: {model_name}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Quantization: {quantization}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # Determine device
        if device == "auto":
            device = get_device()

        # Configure quantization
        quantization_config = None
        if quantization == "4bit":
            logger.info("Configuring 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization == "8bit":
            logger.info("Configuring 8-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        logger.info("Loading model...")
        model_kwargs = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "low_cpu_mem_usage": True,
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = device

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        logger.info(f"✓ Model loaded successfully: {model_name}")

        # Cache the model
        if use_cache:
            _MODEL_CACHE[cache_key] = (model, tokenizer)

        return model, tokenizer

    except ImportError as e:
        logger.error(f"Missing dependency for model loading: {e}")
        raise ImportError(
            f"Failed to import required libraries: {e}\n"
            "Install with: pip install transformers accelerate bitsandbytes"
        )
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    stop_strings: Optional[list] = None
) -> str:
    """
    Generate text response from model

    Args:
        model: Loaded language model
        tokenizer: Tokenizer for the model
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling
        stop_strings: Optional list of strings to stop generation

    Returns:
        Generated text (without the input prompt)
    """
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode output
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the input prompt from response
        response = full_response[len(prompt):].strip()

        # Apply stop strings
        if stop_strings:
            for stop_str in stop_strings:
                if stop_str in response:
                    response = response[:response.index(stop_str)].strip()

        return response

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise RuntimeError(f"Failed to generate response: {e}")


def clear_model_cache():
    """
    Clear the model cache to free memory
    """
    global _MODEL_CACHE

    logger.info(f"Clearing model cache ({len(_MODEL_CACHE)} models)")

    for cache_key in list(_MODEL_CACHE.keys()):
        model, tokenizer = _MODEL_CACHE[cache_key]
        del model
        del tokenizer

    _MODEL_CACHE.clear()

    # Force garbage collection
    import gc
    gc.collect()

    # Clear CUDA cache if available
    if check_cuda_available():
        torch.cuda.empty_cache()

    logger.info("Model cache cleared")


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a model without loading it

    Args:
        model_name: HuggingFace model name

    Returns:
        Dictionary with model information
    """
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name)

        return {
            "model_name": model_name,
            "model_type": config.model_type,
            "vocab_size": config.vocab_size,
            "hidden_size": getattr(config, "hidden_size", None),
            "num_layers": getattr(config, "num_hidden_layers", None),
            "num_attention_heads": getattr(config, "num_attention_heads", None),
        }
    except Exception as e:
        logger.warning(f"Could not get model info: {e}")
        return {"model_name": model_name, "error": str(e)}


def estimate_model_memory(model_name: str, quantization: Optional[str] = None) -> Dict[str, float]:
    """
    Estimate memory requirements for model

    Args:
        model_name: HuggingFace model name
        quantization: Quantization type ('4bit', '8bit', or None)

    Returns:
        Dictionary with memory estimates in GB
    """
    try:
        info = get_model_info(model_name)

        # Rough estimates based on model size
        # These are approximations
        if "7B" in model_name or "7b" in model_name:
            base_size = 14  # GB for FP16
        elif "13B" in model_name or "13b" in model_name:
            base_size = 26  # GB for FP16
        else:
            # Generic estimate based on hidden size
            hidden_size = info.get("hidden_size", 4096)
            num_layers = info.get("num_layers", 32)
            base_size = (hidden_size * num_layers * 12) / 1e9  # Very rough

        # Apply quantization estimates
        if quantization == "4bit":
            model_size = base_size * 0.25
        elif quantization == "8bit":
            model_size = base_size * 0.5
        else:
            model_size = base_size

        return {
            "base_size_gb": base_size,
            "quantized_size_gb": model_size,
            "recommended_vram_gb": model_size * 1.5,  # Account for activations
            "quantization": quantization
        }
    except Exception as e:
        logger.warning(f"Could not estimate memory: {e}")
        return {"error": str(e)}


def test_model_loading(model_name: str = "gpt2") -> bool:
    """
    Test if model loading works with a small model

    Args:
        model_name: Small model to test with (default: gpt2)

    Returns:
        True if loading works
    """
    try:
        logger.info(f"Testing model loading with {model_name}...")
        model, tokenizer = load_model_and_tokenizer(model_name, device="cpu", use_cache=False)

        # Test generation
        response = generate_response(
            model, tokenizer,
            "Hello, this is a test.",
            max_new_tokens=10
        )

        logger.info(f"✓ Model loading test passed")
        logger.info(f"  Test response: {response[:50]}...")

        # Clean up
        del model, tokenizer
        import gc
        gc.collect()

        return True

    except Exception as e:
        logger.error(f"Model loading test failed: {e}")
        return False
