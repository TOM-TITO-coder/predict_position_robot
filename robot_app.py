import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw
import os
import tempfile
import cv2
from torchvision import transforms
import time
import pandas as pd

from ultralytics import YOLO
from model_class import (
    MLPModel, CNN1Model, CNN2Model, MLP_CNN1_Model, MLP_CNN2_Model, CNN1_CNN2_Model, MLP_CNN1_CNN2_Model
)

MODEL_DIR = 'D:/InstituteTechnologyCambodia/Intern-year-5/project/robot_app/models/ex5'
YOLO_PATH = 'D:/InstituteTechnologyCambodia/Intern-year-5/project/robot_app/inferences/last.pt'
IMG_SIZE = (640, 480)
MAX_POLY_POINTS = 50
POLY_DIM = MAX_POLY_POINTS * 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_configs = [
    ('MLP', lambda: MLPModel(POLY_DIM), 'MLP'),
    ('CNN1', CNN1Model, 'CNN1'),
    ('CNN2', CNN2Model, 'CNN2'),
    ('MLP_CNN1', lambda: MLP_CNN1_Model(POLY_DIM), 'MLP_CNN1'),
    ('MLP_CNN2', lambda: MLP_CNN2_Model(POLY_DIM), 'MLP_CNN2'),
    ('CNN1_CNN2', CNN1_CNN2_Model, 'CNN1_CNN2'),
    ('MLP_CNN1_CNN2', lambda: MLP_CNN1_CNN2_Model(POLY_DIM), 'MLP_CNN1_CNN2')
]

@st.cache_resource(show_spinner=False)
def load_yolo(yolo_path):
    return YOLO(yolo_path)

def extract_mask_poly(img_pil, yolo_model, image_size=IMG_SIZE, max_polygon_points=MAX_POLY_POINTS):
    results = yolo_model.predict(np.array(img_pil), verbose=False)[0]
    H, W = results.orig_shape[:2]
    combined = np.zeros((H, W), dtype=np.uint8)
    polys = []
    if results.masks:
        for m, xy in zip(results.masks.data, results.masks.xy):
            m_np = m.cpu().numpy() > 0
            m_resized = np.array(
                Image.fromarray(m_np.astype(np.uint8) * 255).resize((W, H), resample=Image.NEAREST)
            ) > 0
            combined |= m_resized
            polys.append(xy)
    mask_pil = Image.fromarray(combined * 255)
    mask_tensor = transforms.Resize(image_size)(mask_pil)
    mask_tensor = transforms.ToTensor()(mask_tensor)  # [1, H, W]
    poly = polys[0] if len(polys) > 0 else []
    arr = np.asarray(poly, dtype=np.float32) if len(poly) else np.zeros((0,2), dtype=np.float32)
    n = len(arr)
    if n >= max_polygon_points:
        arr = arr[:max_polygon_points]
    else:
        pad = np.zeros((max_polygon_points - n, 2), dtype=np.float32)
        arr = np.vstack([arr, pad])
    poly_tensor = torch.from_numpy(arr)  # (max_pts, 2)
    return mask_tensor, poly_tensor

def draw_polygon_and_pred(img_pil, poly_tensor, pred_xy=None, color="lime"):
    img_draw = img_pil.copy()
    draw = ImageDraw.Draw(img_draw)
    poly_points = poly_tensor.cpu().numpy().squeeze()
    poly_points = poly_points[(poly_points != 0).any(axis=1)]
    if len(poly_points) >= 3:
        poly_points_list = [tuple(map(float, pt)) for pt in poly_points]
        draw.polygon(poly_points_list, outline=color, width=10)
    if pred_xy is not None and len(pred_xy) == 2:
        draw.ellipse(
            [(pred_xy[0] - 6, pred_xy[1] - 6), (pred_xy[0] + 6, pred_xy[1] + 6)],
            fill="red",
            outline="red"
        )
    return img_draw

def predict_xy(model, model_type, img_tensor, mask_tensor, poly_tensor):
    with torch.no_grad():
        if model_type == 'MLP':
            pred = model(poly_tensor)
        elif model_type == 'CNN1':
            pred = model(mask_tensor)
        elif model_type == 'CNN2':
            pred = model(img_tensor)
        elif model_type == 'MLP_CNN1':
            pred = model(mask_tensor, poly_tensor)
        elif model_type == 'MLP_CNN2':
            pred = model(img_tensor, poly_tensor)
        elif model_type == 'CNN1_CNN2':
            pred = model(mask_tensor, img_tensor)
        elif model_type == 'MLP_CNN1_CNN2':
            pred = model(mask_tensor, img_tensor, poly_tensor)
        else:
            return None
        return pred.cpu().numpy().squeeze()

st.set_page_config(page_title="Robot Detection: Image & Video", layout="wide")

st.title("🤖 Object Detection and Localization for Robot 🤖")

# ----------- NEW: File Upload (Image or Video) -----------
file_type = st.sidebar.radio("Select input type", ("Image", "Video"))

uploaded_file = st.file_uploader("Upload an image or video file", type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'])
if uploaded_file is not None:
    if file_type == "Image":
        img_pil = Image.open(uploaded_file).convert("RGB")
        yolo_model = load_yolo(YOLO_PATH)

        # Preprocess and predict for all models
        mask_tensor, poly_tensor = extract_mask_poly(img_pil, yolo_model)
        img_tensor = transforms.Resize(IMG_SIZE)(img_pil)
        img_tensor = transforms.ToTensor()(img_tensor).unsqueeze(0).to(DEVICE)
        mask_tensor_in = mask_tensor.unsqueeze(0).to(DEVICE)
        poly_tensor_in = poly_tensor.unsqueeze(0).to(DEVICE)

        # Predict x, y for all models
        predictions = []
        processed_images = []
        for name, model_class, model_type in model_configs:
            model_path = os.path.join(MODEL_DIR, f"{name}_best.pt")
            model = model_class()
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            pred_xy = predict_xy(model, model_type, img_tensor, mask_tensor_in, poly_tensor_in)
            predictions.append({"Model": name, "x": pred_xy[0] if pred_xy is not None else None, "y": pred_xy[1] if pred_xy is not None else None})
            img_pred = draw_polygon_and_pred(img_pil, poly_tensor, pred_xy)
            processed_images.append(img_pred)

        # Show side-by-side: original and processed from first model
        st.subheader("Original vs Processed Image")
        cols = st.columns(2)
        cols[0].image(img_pil, caption="Original Image")
        cols[1].image(processed_images[0], caption=f"Processed Image ({model_configs[0][0]})")

        # Show table for all model predictions
        st.subheader("Predicted (x, y) for All Models")
        st.table(pd.DataFrame(predictions))
    else:
        # Handle video (same as before but with all models)
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        input_video_path = tfile.name

        frame_skip = st.slider("Show every N frames (higher = faster, less accurate)", 1, 30, 10)
        real_time_fps = st.slider("Frame rate for real-time display", 1, 30, 10)
        yolo_model = load_yolo(YOLO_PATH)

        # Pre-load all models
        loaded_models = []
        for name, model_class, model_type in model_configs:
            model_path = os.path.join(MODEL_DIR, f"{name}_best.pt")
            model = model_class()
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            loaded_models.append((name, model, model_type))

        if st.button("Show Real-Time Side-by-Side Detection (with Table)"):
            cap = cv2.VideoCapture(input_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_placeholder = st.empty()
            table_placeholder = st.empty()
            info_placeholder = st.empty()
            for frame_idx in range(0, total_frames, frame_skip):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                mask_tensor, poly_tensor = extract_mask_poly(img_pil, yolo_model)
                img_tensor = transforms.Resize(IMG_SIZE)(img_pil)
                img_tensor = transforms.ToTensor()(img_tensor).unsqueeze(0).to(DEVICE)
                mask_tensor_in = mask_tensor.unsqueeze(0).to(DEVICE)
                poly_tensor_in = poly_tensor.unsqueeze(0).to(DEVICE)

                predictions = []
                processed_images = []
                for name, model, model_type in loaded_models:
                    pred_xy = predict_xy(model, model_type, img_tensor, mask_tensor_in, poly_tensor_in)
                    predictions.append({"Model": name, "x": pred_xy[0] if pred_xy is not None else None, "y": pred_xy[1] if pred_xy is not None else None})
                    if name == loaded_models[0][0]:
                        img_proc = draw_polygon_and_pred(img_pil, poly_tensor, pred_xy)
                        processed_images.append(img_proc)

                # Show side-by-side
                cols = frame_placeholder.columns(2)
                cols[0].image(img_rgb, caption=f"Original Frame {frame_idx+1}")
                cols[1].image(processed_images[0], caption=f"Processed Frame {frame_idx+1} ({loaded_models[0][0]})")

                # Show prediction table
                table_placeholder.table(pd.DataFrame(predictions))

                info_placeholder.info(f"Frame {frame_idx+1}/{total_frames}")
                time.sleep(1/real_time_fps)
            cap.release()
else:
    st.info("Upload an image or video to get started.")
