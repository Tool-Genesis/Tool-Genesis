from datasets import load_dataset
import os

def download_toucan_dataset():
    dataset_name = "Agent-Ark/Toucan-1.5M"
    configs = ['Kimi-K2', 'OSS', 'Qwen3', 'SFT']
    
    print(f"Starting download of dataset: {dataset_name}")
    
    # Define local path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_local_dir = os.path.join(script_dir, "Toucan-1.5M")
    
    for config in configs:
        print(f"\nProcessing config: {config}")
        try:
            # Load the dataset config
            print(f"Loading {config}...")
            dataset = load_dataset(dataset_name, config)
            
            # Save to a specific subdirectory for this config
            config_dir = os.path.join(base_local_dir, config)
            print(f"Saving {config} to {config_dir}...")
            dataset.save_to_disk(config_dir)
            print(f"Successfully saved {config}.")
            
        except Exception as e:
            print(f"Error processing {config}: {e}")

if __name__ == "__main__":
    download_toucan_dataset()
