import sys
from pathlib import Path
import pandas as pd

# Add the workspace root to sys.path dynamically
root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))

from behaveguard.storage import get_enrolled_profiles
from behaveguard.pipeline import train_user_models

def main():
    data_dir = Path(__file__).resolve().parent
    key_file = data_dir / "key_events.csv"
    mouse_file = data_dir / "mouse_passive.csv"
    
    if not key_file.exists() or not mouse_file.exists():
        print(f"Database files not found at {data_dir}.")
        return
        
    df_keys = pd.read_csv(key_file)
    df_mouse = pd.read_csv(mouse_file)
    
    print(f"Original mouse_passive rows: {len(df_mouse)}")
    
    # Group keys to find max release_ts per session to filter out passive coordinates trailing session completion
    keys_grouped = df_keys.groupby(["subject_id", "collected_at"])["release_ts"].max().reset_index()
    keys_grouped.rename(columns={"release_ts": "max_release_ts"}, inplace=True)
    
    # Merge and filter passive coordinates
    merged = pd.merge(df_mouse, keys_grouped, on=["subject_id", "collected_at"], how="left")
    mask = merged["max_release_ts"].isna() | (merged["ts"] <= merged["max_release_ts"] + 1000)
    
    df_clean = df_mouse[mask].copy()
    print(f"Cleaned mouse_passive rows: {len(df_clean)}")
    
    # Save cleaned data
    df_clean.to_csv(mouse_file, index=False)
    print("Successfully overwrote mouse_passive.csv with cleaned data.")
    
    # Retrain models for all enrolled profiles
    profiles = get_enrolled_profiles()
    print(f"\nRetraining models for profiles: {profiles}")
    for p in profiles:
        print(f"Retraining {p}...")
        train_user_models(p)
    print("All profiles successfully retrained!")

if __name__ == "__main__":
    main()
