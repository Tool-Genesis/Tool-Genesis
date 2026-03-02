from datasets import Dataset, Features, Value
from huggingface_hub import HfApi
import json
import os
import datetime
import argparse

def check_and_create_hf_files(repo_id, file_path, dataset, force_overwrite=False, private=True):
    """
    Checks for README.md and .gitattributes on the Hugging Face Hub and creates them if they don't exist.
    """
    try:
        api = HfApi()
        try:
            repo_files = api.list_repo_files(repo_id, repo_type="dataset")
        except Exception:
            # Repo might not exist, create it.
            api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
            repo_files = []

        # Try to enforce visibility
        try:
            # Newer hub versions
            api.update_repo_visibility(repo_id=repo_id, repo_type="dataset", private=private)
        except Exception:
            try:
                # Fallback to settings
                api.update_repo_settings(repo_id=repo_id, repo_type="dataset", private=private)
            except Exception:
                pass

        if force_overwrite or "README.md" not in repo_files:
            # Create README.md
            features_str = ""
            for key in dataset.features:
                features_str += f"    - name: {key}\n      dtype: string\n"

            num_examples = dataset.num_rows
            num_bytes = dataset.size_in_bytes
            download_size = os.path.getsize(file_path)
            dataset_size = num_bytes

            readme_content = f"""---
dataset_info:
- config_name: default
  features:
{features_str}
  splits:
    - name: train
      num_bytes: {dataset.size_in_bytes}
      num_examples: {len(dataset)}
  download_size: {os.path.getsize(file_path)}
  dataset_size: {dataset.size_in_bytes}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
"""
            api.upload_file(
                path_or_fileobj=readme_content.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
            )
            print("Created or updated README.md")

        if force_overwrite or ".gitattributes" not in repo_files:
            # Create .gitattributes
            gitattributes_content = "*.parquet lfs\n"
            api.upload_file(
                path_or_fileobj=gitattributes_content.encode(),
                path_in_repo=".gitattributes",
                repo_id=repo_id,
                repo_type="dataset",
            )
            print("Created or updated .gitattributes")

    except Exception as e:
        print(f"An error occurred while checking/creating repo files: {e}")

def stringifying_generator(file_path, **kwargs):
    """A generator that reads a JSON file (which is a list of objects) and yields rows with complex values stringified."""
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        upload_time = str(datetime.datetime.now())
        for data in dataset:
            data['upload_timestamp'] = upload_time # Add a unique timestamp to each row
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    data[key] = json.dumps(value, ensure_ascii=False)
                elif value is not None and not isinstance(value, str):
                    data[key] = str(value)
            yield data

def convert_and_upload_to_hf(file_path, repo_id, force_overwrite_files=False, private=True):
    """
    Converts a JSON file to a Parquet dataset and uploads it to the Hugging Face Hub.

    Args:
        file_path (str): The local path to the JSON file.
        repo_id (str): The ID of the repository to upload to (e.g., "username/repo_name").
        force_overwrite_files (bool): Whether to force overwrite README.md and .gitattributes.
    """
    try:
        # Create a dataset directly from the generator.
        # This reads the file line-by-line and applies the string conversion on the fly,
        # making it more memory-efficient than loading the whole file first.
        file_mtime = os.path.getmtime(file_path)
        stringified_dataset = Dataset.from_generator(
            stringifying_generator,
            gen_kwargs={"file_path": file_path, "file_mtime": file_mtime},
        )

        # Check and create essential files on the Hub using the stringified dataset.
        check_and_create_hf_files(repo_id, file_path, stringified_dataset, force_overwrite=force_overwrite_files, private=private)
        
        # Push the dataset to the Hub. It will be stored with the flat schema.
        print(f"Uploading dataset with {stringified_dataset.num_rows} rows ({stringified_dataset.size_in_bytes / 1_000_000:.2f} MB) to {repo_id}")
        commit_message = f"Update dataset at {datetime.datetime.now()}"
        stringified_dataset.push_to_hub(repo_id, commit_message=commit_message)

        print(f"Successfully converted and uploaded {file_path} to {repo_id}")

    except Exception as e:
        print(f"An error occurred: {e}")

def upload_raw_json_file(raw_file_path, repo_id, path_in_repo=None, private=True):
    """
    Uploads a raw JSON file to the Hugging Face Hub dataset repo without conversion.

    Args:
        raw_file_path (str): Local path to the raw JSON file.
        repo_id (str): Target dataset repository ID.
        path_in_repo (str | None): Destination path in the repo; defaults to
            "raw/<basename>" if not provided.
    """
    try:
        if not os.path.exists(raw_file_path):
            print(f"Raw JSON not found: {raw_file_path}. Skipping raw upload.")
            return

        api = HfApi()
        # Ensure repo exists
        try:
            api.list_repo_files(repo_id, repo_type="dataset")
        except Exception:
            api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)

        # Try to enforce visibility
        try:
            api.update_repo_visibility(repo_id=repo_id, repo_type="dataset", private=private)
        except Exception:
            try:
                api.update_repo_settings(repo_id=repo_id, repo_type="dataset", private=private)
            except Exception:
                pass

        if path_in_repo is None:
            path_in_repo = f"raw/{os.path.basename(raw_file_path)}"

        print(f"Uploading raw JSON {raw_file_path} to {repo_id}:{path_in_repo}")
        api.upload_file(
            path_or_fileobj=raw_file_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("Successfully uploaded raw JSON file.")
    except Exception as e:
        print(f"An error occurred during raw JSON upload: {e}")

def upload_raw_files_in_dir(dir_path, repo_id, exclude_names=None, path_prefix="raw/", private=True):
    """
    Upload all files in a directory to the dataset repo as raw files,
    excluding names in exclude_names. Files are placed under path_prefix.

    Args:
        dir_path (str): Directory containing files.
        repo_id (str): Target dataset repository ID.
        exclude_names (set[str] | None): Filenames to exclude.
        path_prefix (str): Prefix in repo for raw files, default "raw/".
    """
    try:
        if not os.path.isdir(dir_path):
            print(f"Directory not found: {dir_path}. Skipping raw dir upload.")
            return

        if exclude_names is None:
            exclude_names = set()

        api = HfApi()
        # Ensure repo exists
        try:
            api.list_repo_files(repo_id, repo_type="dataset")
        except Exception:
            api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)

        # Try to enforce visibility
        try:
            api.update_repo_visibility(repo_id=repo_id, repo_type="dataset", private=private)
        except Exception:
            try:
                api.update_repo_settings(repo_id=repo_id, repo_type="dataset", private=private)
            except Exception:
                pass

        for name in os.listdir(dir_path):
            if name in exclude_names:
                continue
            local_path = os.path.join(dir_path, name)
            if not os.path.isfile(local_path):
                continue
            dest_path = f"{path_prefix}{name}" if path_prefix else name
            print(f"Uploading raw file {local_path} to {repo_id}:{dest_path}")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=dest_path,
                repo_id=repo_id,
                repo_type="dataset",
            )
        print("Successfully uploaded raw directory files.")
    except Exception as e:
        print(f"An error occurred during raw dir upload: {e}")

if __name__ == "__main__":
    # Disable Hugging Face caching for this run
    from datasets import disable_caching
    disable_caching()
    print("Hugging Face datasets caching disabled for this run.")

    # CLI arguments: only expose local file path and repo id per request
    parser = argparse.ArgumentParser(description="Convert a JSON and upload to Hugging Face Hub as a dataset")
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing files; uploads train_data.json as dataset and other files as raw",
    )
    parser.add_argument(
        "--local-file-path",
        default="temp/train-data-gen/train.json",
        help="Local JSON file to upload (default: temp/train-data-gen/train.json)",
    )
    parser.add_argument(
        "--repo-id",
        default="agentgen-v2/test_mcp_server_data_v1",
        help="Target Hugging Face dataset repo id (e.g., username/repo)",
    )
    parser.add_argument(
        "--raw-file-path",
        default="temp/train-data-gen/train_data_raw.json",
        help="Local raw JSON file to upload as-is (default: temp/train-data-gen/train_data_raw.json)",
    )
    parser.add_argument(
        "--raw-path-in-repo",
        default=None,
        help="Path in repo for the raw JSON file (default: raw/<basename>)",
    )
    parser.add_argument(
        "--raw-prefix",
        default="raw/",
        help="Prefix path in repo for raw files when uploading a directory",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the target dataset repository public (default is private)",
    )
    # Keep force overwrite internal unless user needs it; default to True as before
    args = parser.parse_args()

    input_dir = args.input_dir
    local_file_path = args.local_file_path
    hugging_face_repo_id = args.repo_id
    raw_file_path = args.raw_file_path
    raw_path_in_repo = args.raw_path_in_repo
    raw_prefix = args.raw_prefix
    private = not args.public
    force_overwrite = True  # Keep previous behavior

    if input_dir:
        # Derive dataset path from directory
        dataset_path = os.path.join(input_dir, "train_data.json")
        if not os.path.exists(dataset_path):
            alt_path = os.path.join(input_dir, "train.json")
            if os.path.exists(alt_path):
                dataset_path = alt_path
            else:
                print(f"No dataset JSON found in {input_dir} (expected train_data.json or train.json)")
                raise SystemExit(1)

        # Convert and upload dataset
        convert_and_upload_to_hf(dataset_path, hugging_face_repo_id, force_overwrite_files=force_overwrite, private=private)

        # Upload other files in directory as raw
        exclude = {os.path.basename(dataset_path), "README.md", ".gitattributes"}
        upload_raw_files_in_dir(input_dir, hugging_face_repo_id, exclude_names=exclude, path_prefix=raw_prefix, private=private)
    else:
        # Convert and upload the file from explicit path
        convert_and_upload_to_hf(local_file_path, hugging_face_repo_id, force_overwrite_files=force_overwrite, private=private)
        # Upload raw JSON file (as-is) to the same repo
        upload_raw_json_file(raw_file_path, hugging_face_repo_id, path_in_repo=raw_path_in_repo, private=private)