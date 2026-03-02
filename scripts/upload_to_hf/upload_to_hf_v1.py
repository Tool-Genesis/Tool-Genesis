from datasets import Dataset
from huggingface_hub import HfApi
import json
import os
import datetime
import argparse

def check_and_create_hf_files(repo_id, file_path, dataset, force_overwrite=False, private=True):
    try:
        api = HfApi()
        try:
            repo_files = api.list_repo_files(repo_id, repo_type="dataset")
        except Exception:
            api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
            repo_files = []
        try:
            api.update_repo_visibility(repo_id=repo_id, repo_type="dataset", private=private)
        except Exception:
            try:
                api.update_repo_settings(repo_id=repo_id, repo_type="dataset", private=private)
            except Exception:
                pass
        if force_overwrite or "README.md" not in repo_files:
            features_str = ""
            for key in dataset.features:
                features_str += f"    - name: {key}\n      dtype: string\n"
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
        if force_overwrite or ".gitattributes" not in repo_files:
            gitattributes_content = "*.parquet lfs\n"
            api.upload_file(
                path_or_fileobj=gitattributes_content.encode(),
                path_in_repo=".gitattributes",
                repo_id=repo_id,
                repo_type="dataset",
            )
    except Exception as e:
        print(f"An error occurred while checking/creating repo files: {e}")

def stringifying_generator(file_path, limit=None, **kwargs):
    ext = os.path.splitext(file_path)[1].lower()
    upload_time = str(datetime.datetime.now())
    if ext == ".jsonl":
        with open(file_path, 'r', encoding='utf-8') as f:
            count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                data['upload_timestamp'] = upload_time
                for key, value in list(data.items()):
                    if isinstance(value, (dict, list)):
                        data[key] = json.dumps(value, ensure_ascii=False)
                    elif value is not None and not isinstance(value, str):
                        data[key] = str(value)
                yield data
                count += 1
                if isinstance(limit, int) and limit > 0 and count >= limit:
                    break
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            if isinstance(limit, int) and limit > 0:
                dataset = dataset[:limit]
            for data in dataset:
                data['upload_timestamp'] = upload_time
                for key, value in list(data.items()):
                    if isinstance(value, (dict, list)):
                        data[key] = json.dumps(value, ensure_ascii=False)
                    elif value is not None and not isinstance(value, str):
                        data[key] = str(value)
                yield data

def convert_and_upload_to_hf(file_path, repo_id, private=True, force_overwrite_files=True, limit=None):
    try:
        file_mtime = os.path.getmtime(file_path)
        ds = Dataset.from_generator(
            stringifying_generator,
            gen_kwargs={"file_path": file_path, "file_mtime": file_mtime, "limit": limit},
        )
        check_and_create_hf_files(repo_id, file_path, ds, force_overwrite=force_overwrite_files, private=private)
        print(f"Uploading dataset with {ds.num_rows} rows ({ds.size_in_bytes / 1_000_000:.2f} MB) to {repo_id}")
        commit_message = f"Update dataset at {datetime.datetime.now()}"
        ds.push_to_hub(repo_id, commit_message=commit_message)
        print(f"Successfully converted and uploaded {file_path} to {repo_id}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    from datasets import disable_caching
    disable_caching()
    parser = argparse.ArgumentParser(description="Upload a single JSON/JSONL file as a Hugging Face dataset")
    parser.add_argument("--file", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    private = not args.public
    convert_and_upload_to_hf(args.file, args.repo_id, private=private, force_overwrite_files=True, limit=args.limit)