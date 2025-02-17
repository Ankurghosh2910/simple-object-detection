import streamlit as st
import torch
import torchvision.transforms as T
import torchvision
import numpy as np
import cv2
from PIL import Image

@st.cache_resource
def load_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    model.eval()
    model.to(torch.device("cpu"))  
    return model


model = load_model()


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


COLORS = np.random.uniform(0, 255, size=(len(COCO_INSTANCE_CATEGORY_NAMES), 3))


def get_prediction(image, threshold=0.5):
    st.write("Converting image to tensor...")
    transform = T.Compose([T.ToTensor()])
    img = transform(image)  
    st.write(f"Image tensor shape: {img.shape}")  

    st.write("Running model inference...")
    pred = model([img])  

    st.write("Extracting predictions...")
    pred_data = pred[0]

    labels = pred_data['labels'].detach().cpu().numpy()
    boxes = pred_data['boxes'].detach().cpu().numpy()
    scores = pred_data['scores'].detach().cpu().numpy()

    st.write(f"Total objects detected: {len(labels)}")

    # Ensure filtering works correctly
    valid_indices = scores > threshold
    pred_boxes = [((b[0], b[1]), (b[2], b[3])) for b in boxes[valid_indices]]
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in labels[valid_indices]]

    st.write(f"Objects above threshold: {len(pred_boxes)}")

    return pred_boxes, pred_class




def draw_boxes(image, boxes, labels):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    rect_th = max(round(sum(img.shape) / 2 * 0.003), 2)
    text_th = max(rect_th - 1, 1)

    for i in range(len(boxes)):
        p1, p2 = (int(boxes[i][0][0]), int(boxes[i][0][1])), (int(boxes[i][1][0]), int(boxes[i][1][1]))
        color = COLORS[COCO_INSTANCE_CATEGORY_NAMES.index(labels[i])]
        cv2.rectangle(img, p1, p2, color, thickness=rect_th)
        cv2.putText(img, labels[i], (p1[0], p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), text_th)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    return img




st.title("Object Detection App")
st.write("Upload an image and detect objects using Faster R-CNN")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Objects"):
        with st.spinner("Detecting objects..."):
            try:
                model.to(torch.device("cpu"))  
                boxes, labels = get_prediction(image, threshold)
                if not boxes:  
                    st.warning("No objects detected!")
                else:
                    result_image = draw_boxes(image, boxes, labels)
                    st.image(result_image, caption="Detected Objects", use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
