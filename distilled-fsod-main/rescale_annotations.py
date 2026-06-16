import argparse
import json
import os

import PIL
import tqdm

parser = argparse.ArgumentParser(
    description="Create new annotation file where bounding box coordinates " \
        "are scaled to match the target images."
)
parser.add_argument("src_annotation_file")
parser.add_argument("tgt_annotation_file")
parser.add_argument("src_dir")
parser.add_argument("tgt_dir")
args = parser.parse_args()

# check that the input paths are valid
if not os.path.isfile(args.src_annotation_file):
    raise FileNotFoundError(f"No file found at {args.src_annotation_file}")

if os.path.isfile(args.src_dir):
    raise FileNotFoundError(f"Could not find source directory {args.src_dir}")

if os.path.isfile(args.tgt_dir):
    raise FileNotFoundError(f"Could not find target directory {args.tgt_dir}")


# collects the necessary information from the old annotations and groups them
# by their corresponding images
old_annotations = {}
with open(args.src_annotation_file) as json_fp:
    old_annotations = json.load(json_fp)

bboxes = [[] for _ in old_annotations["images"]]

for annotation in old_annotations["annotations"]:
    c_idx = annotation["category_id"]
    new_annotation = {
        "box": annotation["bbox"],
        "category": old_annotations["categories"][c_idx]["name"],
    }
    bboxes[annotation["image_id"]].append(new_annotation)


patches_path = args.tgt_dir

# Create directory for the image patches
for c in old_annotations["categories"]:
    dir_path = os.path.join(patches_path, c["name"])
    if os.path.isdir(dir_path):
        continue
    os.makedirs(dir_path)

cat_cnt = {c["name"]: 0 for c in old_annotations["categories"]}

# Crop the images and save in corresponding directory
for i, annotations in tqdm(enumerate(bboxes)):
    image_name = old_annotations["images"][i]["file_name"]
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
