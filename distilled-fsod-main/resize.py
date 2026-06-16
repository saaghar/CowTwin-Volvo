import argparse
import os
from PIL import Image, ExifTags

parser = argparse.ArgumentParser(
    description="Resize an images or all images in a directory."
)
parser.add_argument("source_path")
parser.add_argument("target_path")
parser.add_argument("width", type=int)
parser.add_argument("height", type=int)
args = parser.parse_args()

if os.path.isfile(args.target_path):
    raise NotADirectoryError("Target path is not a directory.")

is_img = lambda name: name.lower().endswith(
    (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
)

rotation = {1: 0, 8: 90, 3: 180, 6: 270}

if os.path.isfile(args.source_path):
    img = Image.open(args.source_path)
    img = img.resize((args.width, args.height))

    exif = dict(
        (ExifTags.TAGS[k], v) for k, v in img.getexif().items() if k in ExifTags.TAGS
    )
    orientation = exif.get("Orientation")
    if orientation is not None:
        img = img.rotate(rotation[orientation], expand=True)

    name = os.path.split(args.source_path)[-1]
    img.save(os.path.join(args.target_path, name))
else:
    for name in os.listdir(args.source_path):
        if not is_img(name):
            continue
        img = Image.open(os.path.join(args.source_path, name))
        img = img.resize((args.width, args.height))

        exif = dict(
            (ExifTags.TAGS[k], v)
            for k, v in img.getexif().items()
            if k in ExifTags.TAGS
        )
        orientation = exif.get("Orientation")
        if orientation is not None:
            img = img.rotate(rotation[orientation], expand=True)

        img.save(os.path.join(args.target_path, name))
