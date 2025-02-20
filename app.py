import streamlit as st
# Set page config must be the first st command
st.set_page_config(
    page_title="Object Detection App",
    layout="wide"
)

import torch
import torchvision.transforms as T
import torchvision
import numpy as np
import cv2
from PIL import Image

# Initialize session state for model
if 'model' not in st.session_state:
    st.session_state.model = None

@st.cache_resource
def load_model():
    try:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model if not already loaded
if st.session_state.model is None:
    st.session_state.model = load_model()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Generate random colors for each class
COLORS = np.random.uniform(0, 255, size=(len(COCO_INSTANCE_CATEGORY_NAMES), 3))

def get_prediction(image, threshold=0.5):
    transform = T.Compose([T.ToTensor()])
    img = transform(image)
    
    with torch.no_grad():
        pred = st.session_state.model([img])
    
    pred_data = pred[0]
    labels = pred_data['labels'].cpu().numpy()
    boxes = pred_data['boxes'].cpu().numpy()
    scores = pred_data['scores'].cpu().numpy()

    # Filter predictions based on threshold
    valid_indices = scores > threshold
    pred_boxes = [((b[0], b[1]), (b[2], b[3])) for b in boxes[valid_indices]]
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in labels[valid_indices]]
    pred_scores = scores[valid_indices]

    return pred_boxes, pred_class, pred_scores

def draw_boxes(image, boxes, labels, scores):
    # Convert PIL Image to numpy array while preserving quality
    img = np.array(image, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Calculate better scaling factors based on image size
    height, width = img.shape[:2]
    base_size = max(height, width)
    rect_th = max(round(base_size * 0.004), 2)  # Increased thickness
    text_th = max(round(base_size * 0.003), 2)  # Increased text thickness
    font_scale = base_size / 1000.0  # Adjusted font scale
    
    for i in range(len(boxes)):
        p1, p2 = (int(boxes[i][0][0]), int(boxes[i][0][1])), (int(boxes[i][1][0]), int(boxes[i][1][1]))
        color = tuple(map(int, COLORS[COCO_INSTANCE_CATEGORY_NAMES.index(labels[i])]))
        
        # Draw thicker rectangle
        cv2.rectangle(img, p1, p2, color, thickness=rect_th)
        
        # Prepare label text with larger font
        label_text = f"{labels[i]} ({scores[i]:.2f})"
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_th)[0]
        
        # Draw background rectangle for text with padding
        padding = 5
        cv2.rectangle(img, 
                     (p1[0], p1[1] - text_size[1] - 2 * padding), 
                     (p1[0] + text_size[0] + padding, p1[1]), 
                     color, -1)
        
        # Draw text with better positioning
        cv2.putText(img, label_text, 
                    (p1[0] + padding, p1[1] - padding), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                    (255, 255, 255), text_th, cv2.LINE_AA)  # Added LINE_AA for smoother text

    # Convert back to RGB with better quality
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def main():
    st.title("Object Detection App")
    st.write("Upload an image and detect objects using Faster R-CNN")

    # Create two columns with better ratio
    col1, col2 = st.columns([1, 1.5])

    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

    if uploaded_file is not None:
        try:
            # Open image and maintain quality
            image = Image.open(uploaded_file).convert("RGB")
            
            # Display original image with better quality
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Detect Objects"):
                if st.session_state.model is None:
                    st.error("Model failed to load. Please refresh the page.")
                    return
                
                with st.spinner("Detecting objects..."):
                    boxes, labels, scores = get_prediction(image, threshold)
                    
                    if not boxes:
                        st.warning("No objects detected above the confidence threshold!")
                    else:
                        result_image = draw_boxes(image, boxes, labels, scores)
                        with col2:
                            # Display result image with better quality
                            st.image(result_image, caption="Detected Objects", use_container_width=True, channels="RGB")
                            
                            # Display detection summary
                            st.subheader("Detection Summary")
                            objects_count = {}
                            for label, score in zip(labels, scores):
                                if label not in objects_count:
                                    objects_count[label] = {"count": 1, "scores": [score]}
                                else:
                                    objects_count[label]["count"] += 1
                                    objects_count[label]["scores"].append(score)
                            
                            for obj, data in objects_count.items():
                                avg_confidence = np.mean(data["scores"]) * 100
                                st.write(f"- {obj}: {data['count']} instances (avg. confidence: {avg_confidence:.1f}%)")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()

