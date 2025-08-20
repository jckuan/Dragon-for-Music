"""
Google Cloud Storage utilities for DRAGON model
"""
import os
import io
import pandas as pd
import numpy as np
import yaml
import torch
from typing import Union, Optional
from google.cloud import storage
import tempfile
import lmdb


class GCSClient:
    """Google Cloud Storage client for handling file operations"""
    
    def __init__(self):
        self.client = storage.Client()
    
    def parse_gcs_path(self, gcs_path: str):
        """Parse GCS path into bucket name and blob name
        
        Args:
            gcs_path: Path in format 'gs://bucket-name/path/to/file' or 'bucket-name/path/to/file'
            
        Returns:
            tuple: (bucket_name, blob_name)
        """
        if gcs_path.startswith('gs://'):
            gcs_path = gcs_path[5:]  # Remove 'gs://' prefix
        
        parts = gcs_path.split('/', 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ''
        
        return bucket_name, blob_name
    
    def exists(self, gcs_path: str) -> bool:
        """Check if a file exists in GCS"""
        bucket_name, blob_name = self.parse_gcs_path(gcs_path)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.exists()
    
    def read_text(self, gcs_path: str) -> str:
        """Read text file from GCS"""
        bucket_name, blob_name = self.parse_gcs_path(gcs_path)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.download_as_text()
    
    def read_bytes(self, gcs_path: str) -> bytes:
        """Read binary file from GCS"""
        bucket_name, blob_name = self.parse_gcs_path(gcs_path)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.download_as_bytes()
    
    def download_to_file(self, gcs_path: str, local_path: str):
        """Download GCS file to local path"""
        bucket_name, blob_name = self.parse_gcs_path(gcs_path)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)
    
    def upload_from_file(self, local_path: str, gcs_path: str):
        """Upload local file to GCS"""
        bucket_name, blob_name = self.parse_gcs_path(gcs_path)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
    
    def list_blobs(self, gcs_path: str, prefix: str = ''):
        """List blobs in GCS bucket with optional prefix"""
        bucket_name, base_path = self.parse_gcs_path(gcs_path)
        bucket = self.client.bucket(bucket_name)
        full_prefix = os.path.join(base_path, prefix) if base_path else prefix
        return bucket.list_blobs(prefix=full_prefix)


def is_gcs_path(path: str) -> bool:
    """Check if a path is a GCS path"""
    return path.startswith('gs://') or (not path.startswith('/') and '/' in path and not os.path.exists(path))


def get_gcs_file_path(base_path: str, *parts) -> str:
    """Join GCS path components"""
    if base_path.endswith('/'):
        base_path = base_path[:-1]
    
    path = base_path
    for part in parts:
        if part.startswith('/'):
            part = part[1:]
        path = path + '/' + part
    
    return path


def read_csv_from_gcs(gcs_path: str, **kwargs) -> pd.DataFrame:
    """Read CSV file from GCS using pandas"""
    gcs_client = GCSClient()
    content = gcs_client.read_text(gcs_path)
    return pd.read_csv(io.StringIO(content), **kwargs)


def load_numpy_from_gcs(gcs_path: str, **kwargs) -> np.ndarray:
    """Load numpy array from GCS"""
    gcs_client = GCSClient()
    content = gcs_client.read_bytes(gcs_path)
    return np.load(io.BytesIO(content), **kwargs)


def save_numpy_to_gcs(array: np.ndarray, gcs_path: str, **kwargs):
    """Save numpy array to GCS"""
    gcs_client = GCSClient()
    
    # Save to temporary file first
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as temp_file:
        np.save(temp_file.name, array, **kwargs)
        temp_file.flush()
        
        # Upload to GCS
        gcs_client.upload_from_file(temp_file.name, gcs_path)
    
    # Clean up temporary file
    os.unlink(temp_file.name)


def load_yaml_from_gcs(gcs_path: str) -> dict:
    """Load YAML file from GCS"""
    gcs_client = GCSClient()
    content = gcs_client.read_text(gcs_path)
    return yaml.safe_load(content)


def file_exists(path: str) -> bool:
    """Check if file exists (works for both local and GCS paths)"""
    if is_gcs_path(path):
        gcs_client = GCSClient()
        return gcs_client.exists(path)
    else:
        return os.path.isfile(path)


def read_file_content(path: str, mode: str = 'r') -> Union[str, bytes]:
    """Read file content (works for both local and GCS paths)"""
    if is_gcs_path(path):
        gcs_client = GCSClient()
        if 'b' in mode:
            return gcs_client.read_bytes(path)
        else:
            return gcs_client.read_text(path)
    else:
        with open(path, mode) as f:
            return f.read()


def load_torch_from_gcs(gcs_path: str, **kwargs) -> torch.Tensor:
    """Load PyTorch tensor from GCS"""
    gcs_client = GCSClient()
    content = gcs_client.read_bytes(gcs_path)
    return torch.load(io.BytesIO(content), **kwargs)


def save_torch_to_gcs(tensor: torch.Tensor, gcs_path: str, **kwargs):
    """Save PyTorch tensor to GCS"""
    gcs_client = GCSClient()
    
    # Save to temporary file first
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as temp_file:
        torch.save(tensor, temp_file.name, **kwargs)
        temp_file.flush()
        
        # Upload to GCS
        gcs_client.upload_from_file(temp_file.name, gcs_path)
    
    # Clean up temporary file
    os.unlink(temp_file.name)


def create_local_cache_for_lmdb(gcs_lmdb_path: str) -> str:
    """
    Download LMDB database from GCS to local temporary directory
    Returns the local path to the LMDB database
    """
    gcs_client = GCSClient()
    
    # Create temporary directory for LMDB
    temp_dir = tempfile.mkdtemp(prefix='lmdb_cache_')
    
    # List all LMDB files (data.mdb, lock.mdb)
    bucket_name, blob_prefix = gcs_client.parse_gcs_path(gcs_lmdb_path)
    bucket = gcs_client.client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=blob_prefix)
    
    for blob in blobs:
        # Only download LMDB files
        if blob.name.endswith(('.mdb', '.lock')):
            local_file_path = os.path.join(temp_dir, os.path.basename(blob.name))
            blob.download_to_filename(local_file_path)
    
    return temp_dir
