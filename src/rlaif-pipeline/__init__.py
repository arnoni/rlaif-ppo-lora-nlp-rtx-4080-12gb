# src/rlaif_pipeline/__init__.py

"""
RLAIF Pipeline: A package for training language models with AI Feedback.

This package provides the core components for setting up and running RLAIF experiments,
including the main training pipeline, configuration dataclasses, and AI judge implementations.
"""

import logging

# Define the package version.
# This is automatically updated by the 'hatch' build tool based on pyproject.toml.
__version__ = "0.3.0"


# Expose the primary components of the library for easy access.
# This allows users to do `from rlaif_pipeline import main` instead of
# `from rlaif_pipeline.pipeline import main`.
from .pipeline import (
    main,
    RLAIFConfig,
    ModelConfig,
    LoRAConfig,
    TrainingConfig,
    DataConfig,
    EnhancedAIJudge,
    OllamaAIJudge,
)


# Define what symbols are imported when a user does `from rlaif_pipeline import *`.
# This is a best practice for defining a clean public API.
__all__ = [
    # Core function
    "main",

    # Configuration dataclasses
    "RLAIFConfig",
    "ModelConfig",
    "LoRAConfig",
    "TrainingConfig",
    "DataConfig",

    # AI Judge implementations
    "EnhancedAIJudge",
    "OllamaAIJudge",
]


# Set up a null handler for the library's root logger.
# This prevents the library from outputting log messages if the user's
# application has not configured logging, which is a best practice for
# creating well-behaved libraries.
logging.getLogger(__name__).addHandler(logging.NullHandler())