import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import kagglehub
from coco_labels import COCO_INSTANCE_CATEGORY_NAMES

# =====================
# 1. Select Image File using File Dialog
# =====================
def select_image():
    """
    Function to open a file dialog to select an image file.
    Returns:
        str: Path of the selected image.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    return file_path


# =============================
# 2. Load Pre-Trained Faster R-CNN Model
# =============================
def load_model():
    """
    Downloads and loads a pre-trained Faster R-CNN model from TensorFlow Hub.
    Returns:
        tf.saved_model: The loaded Faster R-CNN model.
    """
    # Download the latest version of the model
    model_path = kagglehub.model_download("tensorflow/faster-rcnn-resnet-v1/tensorFlow2/faster-rcnn-resnet50-v1-640x640")
    model = tf.saved_model.load(model_path)  # Load the model
    return model


# ========================
# 3. Image Preprocessing
# ========================
def load_img(image_path):
    """
    Preprocesses the image for Faster R-CNN model.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        np.array: Preprocessed image ready for inference.
    """
    img = Image.open(image_path)
    img = img.convert('RGB')  # Ensure image is in RGB format
    img = img.resize((640, 640))  # Resize to 640x640 (required by the model)
    img = np.array(img)  # Convert to NumPy array
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


# ==========================
# 4. Object Detection Inference
# ==========================
def run_object_detection(model, img):
    """
    Runs object detection on the input image using the pre-trained model.
    
    Args:
        model (tf.saved_model): The pre-trained Faster R-CNN model.
        img (np.array): Preprocessed image for detection.
        
    Returns:
        dict: The detections output from the model.
    """
    input_tensor = tf.convert_to_tensor(img, dtype=tf.uint8)  # Change dtype to tf.uint8
    detections = model(input_tensor)
    return detections


# ==============================
# 5. Draw Bounding Boxes on Image
# ==============================
def draw_detections(image_path, detections, threshold=0.5):
    """
    Draws bounding boxes and labels on the image based on model predictions.
    
    Args:
        image_path (str): Path to the image file.
        detections (dict): Model detections including boxes, classes, and scores.
        threshold (float): Minimum confidence score to display the box.
    """
    # Load image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # Extract detection details
    boxes = detections['detection_boxes'][0].numpy()  # Bounding boxes
    classes = detections['detection_classes'][0].numpy().astype(np.int32)  # Class labels
    scores = detections['detection_scores'][0].numpy()  # Confidence scores

    # Draw bounding boxes for detections above the threshold
    for i in range(boxes.shape[0]):
        if scores[i] >= threshold:
            box = boxes[i]
            y_min, x_min, y_max, x_max = box
            draw.rectangle([x_min * img.width, y_min * img.height, x_max * img.width, y_max * img.height], outline="red", width=2)
            label = COCO_INSTANCE_CATEGORY_NAMES[classes[i]]
            draw.text((x_min * img.width, y_min * img.height), f"{label}: {scores[i]:.2f}", fill="blue")

    # Display the image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


# ==========================
# 6. Running the Object Detection
# ==========================
if __name__ == "__main__":
    # Step 1: Select image using file dialog
    img_path = select_image()

    if img_path:
        # Step 2: Load pre-trained Faster R-CNN model
        model = load_model()

        # Step 3: Load and preprocess the image
        img = load_img(img_path)

        # Step 4: Run object detection
        detections = run_object_detection(model, img)

        # Step 5: Draw bounding boxes on the image
        draw_detections(img_path, detections)
    else:
        print("No image selected.")
