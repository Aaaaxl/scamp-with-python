import tarfile
import os

# Path to the tar file
tar_path = "GSE67835_RAW.tar"
extract_dir = "GSE67835_RAW"

os.makedirs(extract_dir, exist_ok=True)

# Extract
with tarfile.open(tar_path, "r:*") as tar:
    tar.extractall(path=extract_dir)

# List out
extracted_files = os.listdir(extract_dir)
print(f"Extracted {len(extracted_files)} files. Here are the first 10:")
print(extracted_files[:10])
