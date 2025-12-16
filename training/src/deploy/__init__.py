"""
Model Deployment Module

Provides tools for uploading trained models to cloud storage
and updating the model registry in the MakeFour application.
"""

from .upload import upload_model, ModelUploadConfig, StorageBackend
from .registry_update import (
    update_registry,
    generate_model_metadata,
    RegistryConfig,
)

__all__ = [
    "upload_model",
    "ModelUploadConfig",
    "StorageBackend",
    "update_registry",
    "generate_model_metadata",
    "RegistryConfig",
]
