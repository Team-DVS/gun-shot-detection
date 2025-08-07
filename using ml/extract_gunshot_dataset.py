import os
import pandas as pd
import shutil

# Paths
metadata_file = "UrbanSound8K/metadata/UrbanSound8K.csv"
audio_folder = "UrbanSound8K/audio"
output_dir = "dataset"
gunshot_dir = os.path.join(output_dir, "gunshot")
not_gunshot_dir = os.path.join(output_dir, "not_gunshot")

# Create output folders
os.makedirs(gunshot_dir, exist_ok=True)
os.makedirs(not_gunshot_dir, exist_ok=True)

# Load metadata
df = pd.read_csv(metadata_file)

# Filter gunshot and other classes
gunshot_df = df[df["class"] == "gun_shot"]
not_gunshot_df = df[df["class"] != "gun_shot"]

# Balance dataset
not_gunshot_df = not_gunshot_df.sample(n=len(gunshot_df), random_state=42)

# Function to copy files
def copy_files(dataframe, target_dir):
    for _, row in dataframe.iterrows():
        fold = f"fold{row['fold']}"
        file_name = row["slice_file_name"]
        src_path = os.path.join(audio_folder, fold, file_name)
        dst_path = os.path.join(target_dir, file_name)
        shutil.copy(src_path, dst_path)

# Copy files
copy_files(gunshot_df, gunshot_dir)
copy_files(not_gunshot_df, not_gunshot_dir)

print(f"✅ Copied {len(gunshot_df)} gunshot files")
print(f"✅ Copied {len(not_gunshot_df)} non-gunshot files")
