import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms

from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- IMPORT YOUR MODEL CLASSES ---
from model_class import MLPModel, CNN1Model, CNN2Model, MLP_CNN1_Model, MLP_CNN2_Model, CNN1_CNN2_Model, MLP_CNN1_CNN2_Model

# --- SETTINGS ---
MODEL_DIR = 'D:/InstituteTechnologyCambodia/Intern-year-5/project/robot_app/models'
YOLO_PATH = 'D:/InstituteTechnologyCambodia/Intern-year-5/project/robot_app/references/last.pt'
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
            combined |= (m.cpu().numpy() > 0)
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

st.set_page_config(page_title="Robot Position Prediction", layout="wide")
st.title("🤖 Robot Position Prediction (Ablation Study)")

uploaded_file = st.file_uploader("Upload a test image", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    img_pil = Image.open(uploaded_file).convert('RGB')
    st.image(img_pil, caption="Input Image", width=400)

    with st.spinner("Extracting mask and polygon with YOLO..."):
        yolo_model = load_yolo(YOLO_PATH)
        mask_tensor, poly_tensor = extract_mask_poly(img_pil, yolo_model)

    img_tensor = transforms.Resize(IMG_SIZE)(img_pil)
    img_tensor = transforms.ToTensor()(img_tensor).unsqueeze(0).to(DEVICE)
    mask_tensor = mask_tensor.unsqueeze(0).to(DEVICE)
    poly_tensor = poly_tensor.unsqueeze(0).to(DEVICE)

    # Enter actual position for comparison
    st.sidebar.header("Manual Input (Optional)")
    gt_x = st.sidebar.number_input("Ground Truth X", min_value=0.0, max_value=float(IMG_SIZE[0]), value=0.0)
    gt_y = st.sidebar.number_input("Ground Truth Y", min_value=0.0, max_value=float(IMG_SIZE[1]), value=0.0)
    show_gt = st.sidebar.checkbox("Show Actual Position", value=False)

    # Predict with all models
    results = []
    for name, model_class, model_type in model_configs:
        model_path = os.path.join(MODEL_DIR, f"{name}_best.pt")
        if not os.path.exists(model_path):
            st.warning(f"Model file {model_path} not found!")
            continue
        model = model_class()
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
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
                continue
            pred_np = pred.cpu().numpy().squeeze()
        results.append((name, pred_np))

    # Plot predictions
    st.subheader("🔍 Model Predictions (Visualized)")
    cols = st.columns(len(results))
    for idx, (name, pred_np) in enumerate(results):
        with cols[idx]:
            fig, ax = plt.subplots()
            ax.imshow(img_pil)
            ax.scatter(pred_np[0], pred_np[1], color='red', s=60, label='Predicted')
            if show_gt and (gt_x != 0 or gt_y != 0):
                ax.scatter(gt_x, gt_y, color='green', s=60, marker='x', label='Actual')
            ax.set_title(name)
            ax.axis('off')
            ax.legend()
            st.pyplot(fig)

    # Show numerical table
    st.write("### 📋 Model Predictions (x, y)")
    pred_dict = {name: pred for name, pred in results}
    df_pred = pd.DataFrame(pred_dict, index=['x', 'y']).T
    st.dataframe(df_pred)
else:
    st.info("Please upload a test image to start.")
