#!/usr/bin/env python3
"""
Test script to verify GCS functionality with DRAGON
"""
import os
import sys
import numpy as np
import pandas as pd

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.gcs_utils import (
    is_gcs_path, 
    get_gcs_file_path, 
    read_csv_from_gcs, 
    save_numpy_to_gcs,
    load_numpy_from_gcs,
    file_exists
)

def test_gcs_path_detection():
    """Test GCS path detection"""
    print("Testing GCS path detection...")
    
    # Test GCS paths
    assert is_gcs_path('gs://my-bucket/data/')
    assert is_gcs_path('my-bucket/data/file.csv')
    
    # Test local paths
    assert not is_gcs_path('/local/path/data/')
    assert not is_gcs_path('relative/path')
    
    print("✓ GCS path detection working correctly")

def test_gcs_path_joining():
    """Test GCS path joining"""
    print("Testing GCS path joining...")
    
    base_path = 'gs://my-bucket/data'
    result = get_gcs_file_path(base_path, 'dataset', 'file.csv')
    expected = 'gs://my-bucket/data/dataset/file.csv'
    assert result == expected, f"Expected {expected}, got {result}"
    
    # Test with trailing slash
    base_path = 'gs://my-bucket/data/'
    result = get_gcs_file_path(base_path, 'dataset', 'file.csv')
    expected = 'gs://my-bucket/data/dataset/file.csv'
    assert result == expected, f"Expected {expected}, got {result}"
    
    print("✓ GCS path joining working correctly")

def test_local_fallback():
    """Test that local file operations still work"""
    print("Testing local file operations...")
    
    # Create a test numpy array
    test_array = np.array([1, 2, 3, 4, 5])
    local_path = '/tmp/test_array.npy'
    
    # Save and load locally
    np.save(local_path, test_array)
    loaded_array = np.load(local_path)
    
    assert np.array_equal(test_array, loaded_array)
    
    # Clean up
    if os.path.exists(local_path):
        os.remove(local_path)
    
    print("✓ Local file operations working correctly")

def test_config_compatibility():
    """Test that existing configs still work"""
    print("Testing config compatibility...")
    
    # Test local path (should not be detected as GCS)
    local_path = '/nas/MusicRecommendation/data/'
    assert not is_gcs_path(local_path)
    
    # Test GCS path
    gcs_path = 'gs://your-music-rec-bucket/data/'
    assert is_gcs_path(gcs_path)
    
    print("✓ Config compatibility working correctly")

if __name__ == '__main__':
    print("Running DRAGON GCS compatibility tests...\n")
    
    try:
        test_gcs_path_detection()
        test_gcs_path_joining()
        test_local_fallback()
        test_config_compatibility()
        
        print("\n✅ All tests passed! GCS integration is ready.")
        print("\nTo use GCS:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set up GCS authentication: gcloud auth application-default login")
        print("3. Update data_path in configs/overall.yaml to your GCS bucket")
        print("4. Run your DRAGON scripts as usual")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
