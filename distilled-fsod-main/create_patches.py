import argparse
import json
import os

from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description="Creates cropped images of the objects in the annotation file."
)
parser.add_argument("annotation_file")
parser.add_argument("src_dir")
parser.add_argument("dst_dir")
args = parser.parse_args()


# Make sure input paths are valid
if not os.path.isfile(args.annotation_file):
    raise FileNotFoundError(f"No file found at {args.annotation_file}")

if os.path.isfile(args.src_dir):
    raise FileNotFoundError(f"Could not find source directory {args.src_dir}")

if os.path.isfile(args.dst_dir):
    raise FileNotFoundError(f"Could not find destination directory {args.dst_dir}")


# Gather all annotations for each image
all_labels = {}
with open(args.annotation_file) as json_fp:
    all_labels = json.load(json_fp)

bboxes = [[] for _ in all_labels["images"]]

for annotation in all_labels["annotations"]:
    c_idx = annotation["category_id"]
    new_annotation = {
        "box": annotation["bbox"],
        "category": all_labels["categories"][c_idx]["name"],
    }
    bboxes[annotation["image_id"]].append(new_annotation)


patches_path = args.dst_dir

# Create directory for the image patches
for c in all_labels["categories"]:
    dir_path = os.path.join(patches_path, c["name"])
    if os.path.isdir(dir_path):
        continue
    os.makedirs(dir_path)

cat_cnt = {c["name"]: 0 for c in all_labels["categories"]}

# Crop the images and save in corresponding directory
for i, annotations in tqdm(enumerate(bboxes)):
    image_name = all_labels["images"][i]["file_name"]
    image_path = os.path.join(args.src_dir, image_name)
    if not os.path.isfile(image_path):
        continue
    image = Image.open(image_path)
    for annotation in annotations:
        cat = annotation["category"]
        box = annotation["box"]

        left = box[0]
        top = box[1]
        right = left + box[2]
        bottom = top + box[3]

        cropped_image = image.crop((left, top, right, bottom))
        cat_cnt[cat] += 1
        cropped_image.save(os.path.join(patches_path, cat, f"{cat_cnt[cat]:05}.png"))
