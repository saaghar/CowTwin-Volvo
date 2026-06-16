import argparse
import os
import shutil
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Creates a directory with a flat file structure."
)
parser.add_argument("src_dir")
parser.add_argument("dst_dir")
args = parser.parse_args()


# check input paths
if os.path.isfile(args.src_dir):
    raise FileNotFoundError(f"Could not find source directory {args.src_dir}")

if os.path.isfile(args.dst_dir):
    raise FileNotFoundError(f"Could not find destination directory {args.dst_dir}")


# goes through the dir to flatten it, renaming the files as they are copied
labels = os.listdir(args.src_dir)

for label in tqdm(labels):
    label_path = os.path.join(args.src_dir, label) 
    if os.path.isfile(label_path):
        continue

    image_names = os.listdir(label_path)
    for name in image_names:    # names are assumed to just be numbers
        im_path = os.path.join(label_path, name)
        if not os.path.isfile(im_path):
            continue

        dst_path = os.path.join(args.dst_dir, f"{label}-{name}")
        shutil.copyfile(im_path, dst_path)