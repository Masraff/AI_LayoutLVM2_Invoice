from transformers import LayoutLMv2Processor
from PIL import Image
import torch
import json
import os

# Directories
labeled_data_dir = 'labeled_data_normalized'
images_dir = 'images'
encoded_data_dir = 'encoded_data'
os.makedirs(encoded_data_dir, exist_ok=True)

# Load LayoutLMv2 processor
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", apply_ocr=False)

def adjust_boxes(boxes):
    adjusted_boxes = []
    for box in boxes:
        left, top, right, bottom = box
        # Ensure left < right and top < bottom
        if left > right:
            left, right = right, left
        if top > bottom:
            top, bottom = bottom, top
        adjusted_boxes.append([left, top, right, bottom])
    return adjusted_boxes

# Iterate through labeled data files
for filename in os.listdir(labeled_data_dir):
    if filename.endswith('.json'):
        # Load the JSON data and image
        json_path = os.path.join(labeled_data_dir, filename)
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        image_id = filename.split('.')[0]
        image_path = os.path.join(images_dir, f'{image_id}.jpg')
        image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format


        # Extract words, bounding boxes, and labels from JSON
        def parse_json(json_data):
            words = []
            boxes = []
            labels = []
            for entry in json_data:
                words.append(entry['text'])
                boxes.append(entry['boundingBox'])
                labels.append(entry['label'])
            return words, boxes, labels


        words, boxes, labels = parse_json(json_data)
        boxes = adjust_boxes(boxes)

        # Check if there are words to process
        if len(words) == 0:
            print(f"Skipping {filename} as no words are present.")
            continue

        max_length = 512
        words = words[:max_length]
        boxes = boxes[:max_length]
        labels = labels[:max_length]
        # Create encodings
        encoding = processor(images=image, text=words, boxes=boxes, return_tensors="pt", padding=True,
                             truncation=True, max_length = 512)

        # Convert labels to tensor
        labels_tensor = torch.tensor(labels).unsqueeze(0)

        # Add labels to encoding
        data = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'token_type_ids': encoding['token_type_ids'],
            'bbox': encoding['bbox'],
            'image': encoding['image'],  # Update to 'image'
            'labels': labels_tensor
        }

        # Save encoded data
        encoded_data_path = os.path.join(encoded_data_dir, f'{image_id}.pt')
        torch.save(data, encoded_data_path)

print("All data has been encoded and saved.")
