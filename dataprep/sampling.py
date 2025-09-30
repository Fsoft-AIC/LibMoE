import json
from tqdm import tqdm
import random
from typing import Dict, List, Any
from pathlib import Path

# Constants
DATA_DIR = Path("/cm/archive/anonymous/data/jsons/onevison600K")
INPUT_FILE = DATA_DIR / "onevision_single_img_standard.json"
OUTPUT_FILE = DATA_DIR / "onevision_single_img_standard_sampled.json"
STATS_OVERALL_FILE = DATA_DIR / "cag_stats_overall.json"
STATS_SAMPLED_FILE = DATA_DIR / "cag_stats_sampled.json"
STATS_SAMPLED_TXT = DATA_DIR / "cag_stats_sampled.txt"

def load_data(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSON data from file."""
    with open(file_path, "r") as f:
        return json.load(f)

def group_by_source(data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group data by data_source."""
    cag = {}
    for item in tqdm(data, desc="Processing"):
        source = item['data_source']
        if source not in cag:
            cag[source] = []
        cag[source].append(item)
    return cag

def save_stats(stats: Dict[str, int], file_path: Path) -> None:
    """Save statistics to a JSON file."""
    with open(file_path, "w") as f:
        json.dump(stats, f, indent=4)

def sample_data(cag: Dict[str, List[Dict[str, Any]]], sample_ratio: float = 0.5) -> Dict[str, List[Dict[str, Any]]]:
    """Sample data from each category."""
    sampled_data = {}
    for key, value in cag.items():
        sample_size = int(len(value) * sample_ratio)
        sampled_data[key] = random.sample(value, sample_size)
    return sampled_data

def main():
    # Load and process data
    data = load_data(INPUT_FILE)
    cag_org = group_by_source(data)
    # remove cag multi image
    with open("/cm/archive/anonymous/data/jsons/onevison600K/cag_selected.json", "r") as f:
        cag_selected = json.load(f)
    
    cag = {}
    for k,v in cag_org.items():
        if k not in cag_selected: continue
        cag[k] = v
    # Save overall statistics
    cag_stats = {key: len(value) for key, value in cag.items()}
    save_stats(cag_stats, STATS_OVERALL_FILE)
    
    # Sample data
    sampled_data = sample_data(cag, sample_ratio= 0.25)
    final_data = []
    for samples in sampled_data.values():
        final_data.extend(samples)
    
    # Save sampled statistics
    sampled_stats = {key: len(value) for key, value in sampled_data.items()}
    save_stats(sampled_stats, STATS_SAMPLED_FILE)
    
    # Log total categories
    with open(STATS_SAMPLED_TXT, "a") as f:
        f.write(f"Total sampled categories after sampling 50%: {len(final_data)}\n")
    
    # Shuffle and save final data
    random.shuffle(final_data)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_data, f, indent=4)

if __name__ == "__main__":
    main()