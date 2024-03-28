import os
import imgaug.augmenters as iaa
import imageio
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import time


seq = iaa.Sequential([
    iaa.Fliplr(0.1),  # Horizontal flip
    iaa.Flipud(0.1),  # Vertical flip 
    iaa.GaussianBlur(sigma=(0, 1.0)),  # Blur images with a sigma of 0 to 1.0
], random_order=True)

def load_bounding_boxes(label_path):
    bounding_boxes = []
    with open(label_path, 'r') as file:
        for line in file.readlines():
            class_id, cx, cy, w, h = map(float, line.split())
            bounding_boxes.append(BoundingBox(x1=cx-w/2, y1=cy-h/2, x2=cx+w/2, y2=cy+h/2, label=class_id))
    return bounding_boxes

def save_bounding_boxes(bounding_boxes, label_path, image_width, image_height):
    with open(label_path, 'w') as file:
        for bb in bounding_boxes:
            # Convert from absolute coordinates to YOLO format (normalized cx, cy, w, h)
            cx = (bb.x1 + bb.x2) / 2 / image_width
            cy = (bb.y1 + bb.y2) / 2 / image_height
            w = (bb.x2 - bb.x1) / image_width
            h = (bb.y2 - bb.y1) / image_height
            # Write to file in the format: class_id cx cy w h
            file.write(f"{int(bb.label)} {cx} {cy} {w} {h}\n")

def get_unique_filename(original_filename):
    # using a timestamp or an incrementing number as a unique identifier
    timestamp = int(time.time())
    name, ext = os.path.splitext(original_filename)
    unique_filename = f"{name}_aug_{timestamp}{ext}"
    return unique_filename


# Paths to the data based on config.yaml
images_path = "C:/Users/lotus/Desktop/morring/code/data/images/train"
labels_path = "C:/Users/lotus/Desktop/morring/code/data/labels/train"
augmented_images_path = "C:/Users/lotus/Desktop/morring/code/data/images/train_augmented"
augmented_labels_path = "C:/Users/lotus/Desktop/morring/code/data/labels/train_augmented"

# Making sure output directories exist
os.makedirs(augmented_images_path, exist_ok=True)
os.makedirs(augmented_labels_path, exist_ok=True)

# List of image files
image_files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]

for image_file in image_files:
    # Load the image
    image = imageio.imread(os.path.join(images_path, image_file))
    
    # Load the bounding boxes
    label_file = os.path.splitext(image_file)[0] + '.txt'  # Change to your label file extension
    bbs = load_bounding_boxes(os.path.join(labels_path, label_file))

    # Convert to imgaug BoundingBoxesOnImage object
    bbs_on_image = BoundingBoxesOnImage(bbs, shape=image.shape)
    
    # Augment the image and the bounding boxes
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs_on_image)
    # Generate a unique filename for the augmented image and label
    unique_image_filename = get_unique_filename(image_file)
    unique_label_filename = get_unique_filename(label_file)
    # Save the augmented image
    imageio.imwrite(os.path.join(augmented_images_path, unique_image_filename), image_aug)    
    # Remove any bounding boxes that have fallen outside of the image after the augmentation
    bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
    
    # Save the augmented bounding boxes
    save_bounding_boxes(bbs_aug, os.path.join(augmented_labels_path, unique_label_filename), image_aug.shape[1], image_aug.shape[0])
    
print("Data augmentation is complete.")

