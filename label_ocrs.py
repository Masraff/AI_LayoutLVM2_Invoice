import os
import json
from shapely.geometry import Point, box

# Define directories
label_studio_path = 'label_studio.json'
ocr_data_dir = 'dataset_bbox_not_normalized'
output_dir = 'labeled_data_normalized'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Step 1: Load Label Studio annotations
with open(label_studio_path, 'r') as f:
    label_studio_data = json.load(f)

# Extract bounding boxes for total amounts
total_amount_boxes = {}
for entry in label_studio_data:
    expense_id = entry['file_upload'].split('.')[0].split('-')[1]  # Assuming file names match expense_id
    annotations = entry['annotations'][0]['result']
    total_amount_boxes[expense_id] = []

    for annotation in annotations:
        if annotation['value']['rectanglelabels'][0] == 'Total Amount':
            # Get normalized coordinates
            x_norm = annotation['value']['x']
            y_norm = annotation['value']['y']
            width_norm = annotation['value']['width']
            height_norm = annotation['value']['height']

            # Convert normalized coordinates to pixel values
            image_width = annotation['original_width']
            image_height = annotation['original_height']

            x_pixel = x_norm / 100 * image_width
            y_pixel = y_norm / 100 * image_height
            width_pixel = width_norm / 100 * image_width
            height_pixel = height_norm / 100 * image_height

            # Store the bounding box in pixel values
            total_amount_boxes[expense_id].append({
                'x': x_pixel,
                'y': y_pixel,
                'width': width_pixel,
                'height': height_pixel
            })

print("Total amount bounding boxes extracted.")

# Step 2: Load OCR bounding boxes and label each word
def load_ocr_bboxes(expense_id):
    with open(f'{ocr_data_dir}/{expense_id}.json', 'r') as f:
        return json.load(f)

def is_inside_total_box(word_bbox, total_boxes):
    word_center = Point(
        (word_bbox[0] + word_bbox[2]) / 2,
        (word_bbox[1] + word_bbox[3]) / 2
    )
    for tb in total_boxes:
        total_box = box(tb['x'], tb['y'], tb['x'] + tb['width'], tb['y'] + tb['height'])
        if total_box.contains(word_center):
            print("in")
            return True
    print("not")
    return False

# Step 3: Process each expense, label words, and save to output directory
for expense_id, total_boxes in total_amount_boxes.items():
    try:
        ocr_bboxes = load_ocr_bboxes(expense_id)
    except FileNotFoundError:
        print(f"OCR file for {expense_id} not found, skipping.")
        continue

    labeled_words = []
    for word_data in ocr_bboxes:
        word_bbox = word_data['BBox']
        is_total = is_inside_total_box(word_bbox, total_boxes)


        labeled_words.append({
            'text': word_data['Word'],
            'boundingBox': word_bbox,
            'label': 1 if is_total else 0
        })

    # Save labeled data for each expense
    output_file = os.path.join(output_dir, f'{expense_id}_labeled.json')
    with open(output_file, 'w') as f:
        json.dump(labeled_words, f, indent=4)
    print(f"Labeled data saved for {expense_id} at {output_file}")

print("All files processed and labeled.")
