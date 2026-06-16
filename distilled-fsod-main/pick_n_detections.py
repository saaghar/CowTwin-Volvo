import json
from collections import Counter, defaultdict
import os

def count_and_display_coco_instances(coco_data, title="COCO Annotation Instance Counts"):

    category_map = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
    category_ids_counts = Counter()

    for ann in coco_data.get('annotations', []):
        if 'category_id' in ann:
            category_ids_counts[ann['category_id']] += 1
        else:
            print("Warning: Found an annotation without a 'category_id'. Skipping it.")

    print(f"\n--- {title} ---")
    print("---------------------------------------")
    max_name_len = max((len(name) for name in category_map.values()), default=20)
    max_id_len = max((len(str(id)) for id in category_map.keys()), default=5)

    found_any_counts = False
    for category_id, category_name in sorted(category_map.items(), key=lambda item: item[1]):
        count = category_ids_counts.get(category_id, 0)
        if count > 0:
            found_any_counts = True
            print(f"{category_name:<{max_name_len}} (ID: {str(category_id):<{max_id_len}}) : {count} instances")

    if not found_any_counts and category_map: # Check if category_map is not empty
        print("No instances found for any of the defined categories.")
    elif not category_map and not category_ids_counts: # Neither categories nor annotations
        print("No categories or annotations defined in the file.")
    print("---------------------------------------")
    return True


def create_n_shot_coco(input_json_path, output_json_path, n_per_category):
    """
    Creates a new COCO JSON file with N instances per category.

    Args:
        input_json_path (str): Path to the original COCO JSON file.
        output_json_path (str): Path to save the new N-shot COCO JSON file.
        n_per_category (int): The maximum number of instances to keep per category.
    """
    print(f"\nLoading annotations from: {input_json_path}")
    with open(input_json_path, 'r') as f:
        coco_data = json.load(f)

    print(f"Original instance counts:")
    if not count_and_display_coco_instances(coco_data, "Original Instance Counts"):
        return # Stop if original data is malformed

    annotations = coco_data.get('annotations', [])
    categories = coco_data.get('categories', [])
    images_original = coco_data.get('images', [])
    info = coco_data.get('info', {})
    licenses = coco_data.get('licenses', [])

    # Group annotations by category_id
    annotations_by_category = defaultdict(list)
    for ann in annotations:
        annotations_by_category[ann['category_id']].append(ann)

    selected_annotations = []
    print(f"\nSelecting up to {n_per_category} instances per category...")

    # Create category_id to name mapping for logging
    category_map = {cat['id']: cat['name'] for cat in categories}

    for category_id, anns in annotations_by_category.items():
        category_name = category_map.get(category_id, f"ID {category_id}")
        if len(anns) > n_per_category:
            print(f"  Category '{category_name}': Found {len(anns)} instances, selecting {n_per_category}.")
            selected_annotations.extend(anns[:n_per_category])
        else:
            print(f"  Category '{category_name}': Found {len(anns)} instances, selecting all.")
            selected_annotations.extend(anns)

    if not selected_annotations:
        print("\nNo annotations were selected. The output file will be empty or minimal.")
        # Create a minimal valid COCO structure if no annotations are selected
        new_coco_data = {
            'info': info,
            'licenses': licenses,
            'images': [],
            'annotations': [],
            'categories': categories
        }
    else:
        # Get the set of image_ids that are present in the selected annotations
        image_ids_with_selected_annotations = {ann['image_id'] for ann in selected_annotations}

        # Filter the original images list
        selected_images = [img for img in images_original if img['id'] in image_ids_with_selected_annotations]

        print(f"\nTotal selected annotations: {len(selected_annotations)}")
        print(f"Total images with selected annotations: {len(selected_images)}")

        new_coco_data = {
            'info': info,
            'licenses': licenses,
            'images': selected_images,
            'annotations': selected_annotations,
            'categories': categories  # Keep all original categories
        }

    print(f"\nSaving N-shot annotations to: {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(new_coco_data, f, indent=4)
    print("N-shot file created successfully.")

    print(f"\nNew N-shot instance counts in '{output_json_path}':")
    count_and_display_coco_instances(new_coco_data, "N-Shot Instance Counts")


if __name__ == '__main__':
    # --- Configuration ---
    input_annotation_file = 'distilled_labels_03_5shot_inc.json'

    with open(input_annotation_file, 'r') as f:
        input_data = json.load(f)
    count_and_display_coco_instances(input_data)
    # Name for the output N-shot file. It will be saved in the same directory.
    output_n_shot_file_template = 'distilled_labels_03_5shot_{N}pick.json'
    # -----------------

    if not os.path.exists(input_annotation_file):
        print(f"Error: The input annotation file '{input_annotation_file}' was not found.")
        print("Please make sure the file exists or update the 'input_annotation_file' variable in the script.")
    else:
        while True:
            try:
                n_value_str = input("Enter the number of instances (N) to keep per category (e.g., 1, 5, 10): ")
                n_value = int(n_value_str)
                if n_value <= 0:
                    print("N must be a positive integer. Please try again.")
                else:
                    break
            except ValueError:
                print("Invalid input. Please enter an integer.")

        # Construct output file name based on N
        output_n_shot_file = output_n_shot_file_template.replace("{N}", str(n_value))

        create_n_shot_coco(input_annotation_file, output_n_shot_file, n_value)