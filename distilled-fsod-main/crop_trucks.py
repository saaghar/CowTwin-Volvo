import os
import shutil
import cv2
from math import floor, ceil
from ultralytics import YOLO
from tqdm import tqdm

model_path = "weights/truck_cabin.pt"

src_dir = "inference_images/horn_sd35large"
dst_dir = "inference_images/test"

if not os.path.isdir(src_dir):
    raise FileNotFoundError("Source directory not found.")

if os.path.isdir(dst_dir):
    shutil.rmtree(dst_dir)
os.mkdir(dst_dir)

model = YOLO(model_path)
valid_extensions = {'.jpg', '.jpeg', '.png'}

images = []
im_names = os.listdir(src_dir)
for name in im_names:
    im_path = os.path.join(src_dir, name)
    if any(im_path.lower().endswith(ext) for ext in valid_extensions):
        im = cv2.imread(im_path)
        images.append(im)
    else:
        im_names.remove(name)
        print(f"INVALID IMAGE PATH: {im_path}. REMOVING THIS FILE!")

batch_size = 250

for i in tqdm(range(0, len(images), batch_size), desc="Processing images"):
    results = model(images[i:i+batch_size], verbose=False)

    for result, image, name in zip(results, images[i:i+batch_size], im_names[i:i+batch_size]):
        boxes = result.boxes  # Boxes object for bounding box outputs
        if boxes.xyxy.shape[0] < 1:
            continue
        x1, y1, x2, y2 = tuple(boxes.xyxy[0,:].to('cpu').tolist())
        cropped_image = image[floor(y1):ceil(y2), floor(x1):ceil(x2)]
        try:
            cv2.imwrite(os.path.join(dst_dir, name), cropped_image)
        except Exception as e:
                print(f"Error processing {os.path.join(dst_dir, name)}: {e}")
            
# Print number of saved images
files = 0

for root, dirnames, filenames in os.walk(dst_dir):
    files += len(filenames)
    
print(f"Detected {files} truck cabins out of {len(images)}!")