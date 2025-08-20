# Google Cloud Storage (GCS) Setup Guide

This guide explains how to configure DRAGON to work with data stored in Google Cloud Storage buckets.

## Prerequisites

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Google Cloud authentication:
   - Install the Google Cloud SDK
   - Run `gcloud auth application-default login`
   - Or set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to your service account key file

## Configuration

### 1. Update data_path in configs/overall.yaml

Change the `data_path` in your configuration file to point to your GCS bucket:

```yaml
# For GCS bucket
data_path: 'gs://your-bucket-name/path/to/data/'

# Or without gs:// prefix
data_path: 'your-bucket-name/path/to/data/'
```

### 2. Data Structure in GCS

Your GCS bucket should maintain the same directory structure as local storage:

```
gs://your-bucket-name/path/to/data/
├── dataset1/
│   ├── inter_file.inter
│   ├── text_file.txt
│   ├── img_dir/
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── vision_features.npy
│   ├── text_features.npy
│   └── user_graph_dict.npy
├── dataset2/
│   └── ...
```

## Supported Operations

The following file operations now work with both local and GCS paths:

- Reading CSV files (pandas)
- Loading/saving NumPy arrays
- Loading/saving PyTorch tensors
- Reading YAML configuration files
- Accessing LMDB databases (automatically cached locally)

## Usage Examples

### Running the u-u matrix generation tool:
```bash
cd tools
python generate-u-u-matrix.py --dataset sports
```

### Training with GCS data:
```bash
python main.py --dataset sports --config_files configs/overall.yaml configs/dataset/sports.yaml
```

## Performance Considerations

1. **LMDB Databases**: Image databases are automatically downloaded and cached locally for better performance.

2. **Caching**: Consider implementing local caching for frequently accessed files to reduce GCS API calls.

3. **Network**: Ensure stable internet connection for GCS operations.

## Troubleshooting

### Authentication Issues
```bash
# Set up default credentials
gcloud auth application-default login

# Or use service account
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
```

### Permission Issues
Ensure your GCS credentials have the following permissions:
- `storage.objects.get` (read files)
- `storage.objects.create` (write files)
- `storage.objects.list` (list files)

### Path Format
- Use forward slashes `/` in GCS paths
- Paths can start with `gs://` or just the bucket name
- Ensure consistent trailing slashes in directory paths
