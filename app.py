import os
import time
import tempfile
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import streamlit as st
import torch
import plotly.graph_objects as go

from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
)

# ============================================================
# 0. Streamlit 기본 설정
# ============================================================
st.set_page_config(
    page_title="MONAI 3D Segmentation Demo",
    page_icon="🧠",
    layout="wide",
)

st.title("MONAI + Streamlit 3D 의료영상 데모")
st.caption("NIfTI(.nii / .nii.gz) 파일을 업로드하거나 폴더에서 선택해 3D segmentation 추론 결과를 확인합니다.")

# ============================================================
# 1. Session State 초기화
# ============================================================
if "inference_done" not in st.session_state:
    st.session_state.inference_done = False

if "result" not in st.session_state:
    st.session_state.result = None

if "raw_shape" not in st.session_state:
    st.session_state.raw_shape = None

if "source_name" not in st.session_state:
    st.session_state.source_name = None

if "source_token" not in st.session_state:
    st.session_state.source_token = None


# ============================================================
# 2. 공통 유틸 함수
# ============================================================
def is_nifti_file_name(file_name: str) -> bool:
    """파일명이 .nii 또는 .nii.gz 인지 확인"""
    lower = file_name.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def list_nifti_files(folder_path: str, recursive: bool = False) -> List[str]:
    """
    폴더 내 .nii / .nii.gz 파일 목록 반환
    """
    folder = Path(folder_path)
    if not folder.exists(): return []
    if not folder.is_dir(): return []
    candidates = folder.rglob("*") if recursive else folder.iterdir()
    files = [str(p.resolve()) for p in candidates if p.is_file() and is_nifti_file_name(p.name)]
    files.sort()
    return files


def make_source_token_from_path(file_path: str) -> str:
    p = Path(file_path)
    stat = p.stat()
    return f"path::{str(p.resolve())}::{stat.st_size}::{stat.st_mtime}"


def make_source_token_from_upload(uploaded_file) -> str:
    return f"upload::{uploaded_file.name}::{uploaded_file.size}"


def generate_synthetic_nifti():
    """테스트용 가상 3D NIfTI 파일 생성 (구 형태)"""
    size = (96, 96, 96)
    data = np.zeros(size, dtype=np.float32)
    z, y, x = np.ogrid[:size[0], :size[1], :size[2]]
    center = (48, 48, 48)
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    data[dist_from_center <= 25] = 100.0
    data += np.random.normal(-50, 20, size)
    affine = np.eye(4)
    image = nib.Nifti1Image(data, affine)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        nib.save(image, tmp.name)
        return tmp.name


def get_available_device_options():
    options = []
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        options.append(("mps", "GPU (Apple Silicon / MPS)"))
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            options.append((f"cuda:{i}", f"GPU (CUDA:{i} / {torch.cuda.get_device_name(i)})"))
    options.append(("cpu", "CPU"))
    return options


def build_model() -> UNet:
    return UNet(
        spatial_dims=3, in_channels=1, out_channels=2,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
        num_res_units=2, norm=Norm.BATCH,
    )


@st.cache_resource
def load_model_cached(model_path: str, device_name: str):
    device = torch.device(device_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    model = build_model()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def reset_inference_state():
    st.session_state.inference_done = False
    st.session_state.result = None
    st.session_state.raw_shape = None
    st.session_state.source_name = None
    for key in list(st.session_state.keys()):
        if key.startswith("slice_idx_axis_"):
            del st.session_state[key]


def save_uploaded_file_to_temp(uploaded_file) -> str:
    lower_name = uploaded_file.name.lower()
    suffix = ".nii.gz" if lower_name.endswith(".nii.gz") else ".nii" if lower_name.endswith(".nii") else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def preprocess_case(image_path: str) -> Dict:
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear",)),
        ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image"),
        EnsureTyped(keys=["image"]),
    ])
    data = {"image": image_path}
    data = transforms(data)
    return data


def extract_affine_from_meta(image_tensor) -> np.ndarray:
    if hasattr(image_tensor, "affine") and image_tensor.affine is not None:
        affine = image_tensor.affine
        try: affine = affine.cpu().numpy()
        except: affine = np.array(affine)
        if affine.ndim == 3: affine = affine[0]
        return affine
    return np.eye(4, dtype=float)


def run_inference(data: Dict, model, device_name: str, roi_size=(160, 160, 160), sw_batch_size=1) -> Dict:
    device = torch.device(device_name)
    image = data["image"]
    affine = extract_affine_from_meta(image)
    input_tensor = image.unsqueeze(0).to(device)
    start_time = time.time()
    with torch.no_grad():
        logits = sliding_window_inference(inputs=input_tensor, roi_size=roi_size, sw_batch_size=sw_batch_size, predictor=model)
    elapsed = time.time() - start_time
    raw_pred = torch.argmax(logits, dim=1)
    raw_pred_np = raw_pred[0].cpu().numpy().astype(np.uint8)
    image_np = image[0].cpu().numpy()
    return {"image_np": image_np, "raw_pred_np": raw_pred_np, "affine": affine, "elapsed": elapsed}


def visualize_3d_mask(mask_np):
    """Plotly를 이용한 3D Segmentation 마스크 시각화"""
    try:
        step = 2
        sub_mask = mask_np[::step, ::step, ::step]
        if sub_mask.sum() == 0: return None
        z, y, x = np.where(sub_mask > 0)
        fig = go.Figure(data=[go.Isosurface(
            x=x.flatten(), y=y.flatten(), z=z.flatten(),
            value=np.ones_like(x).flatten(),
            isomin=0.5, isomax=1.5, opacity=0.6,
            colorscale='Reds', caps=dict(x_show=False, y_show=False)
        )])
        fig.update_layout(
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
            margin=dict(l=0, r=0, b=0, t=0),
            height=600
        )
        return fig
    except Exception as e:
        st.error(f"3D 시각화 생성 중 에러 발생: {e}")
        return None


def extract_slice(volume, axis, index):
    if axis == 0: return volume[index, :, :]
    elif axis == 1: return volume[:, index, :]
    else: return volume[:, :, index]


def get_best_slice(mask_3d, axis):
    if mask_3d.sum() == 0: return mask_3d.shape[axis] // 2
    slice_sums = [extract_slice(mask_3d, axis, i).sum() for i in range(mask_3d.shape[axis])]
    return int(np.argmax(slice_sums))


def make_overlay_figure(image_slice, pred_slice):
    mn, mx = image_slice.min(), image_slice.max()
    img_disp = (image_slice - mn) / (mx - mn) if mx - mn > 1e-8 else np.zeros_like(image_slice)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_disp, cmap="gray"); axes[0].axis("off"); axes[0].set_title("Image")
    axes[1].imshow(pred_slice, cmap="gray"); axes[1].axis("off"); axes[1].set_title("Prediction")
    axes[2].imshow(img_disp, cmap="gray"); axes[2].imshow(pred_slice, cmap="Reds", alpha=0.35)
    axes[2].axis("off"); axes[2].set_title("Overlay")
    plt.tight_layout()
    return fig


# ============================================================
# 3. 사이드바 및 모델 로드
# ============================================================
with st.sidebar:
    st.header("실행 설정")
    model_path = st.text_input("모델 파일 경로", value="./best_metric_model.pth")
    device_items = get_available_device_options()
    device_name = st.selectbox("디바이스", options=[v for v, _ in device_items], format_func=lambda x: dict(device_items)[x])
    roi_x = st.number_input("ROI X", 32, 256, 160)
    roi_y = st.number_input("ROI Y", 32, 256, 160)
    roi_z = st.number_input("ROI Z", 32, 256, 160)

try:
    model = load_model_cached(model_path, device_name)
    st.sidebar.success("모델 로드 성공")
except Exception as e:
    st.sidebar.error(f"모델 로드 실패: {e}")
    st.stop()

# ============================================================
# 5. 입력 파일 선택
# ============================================================
st.subheader("입력 파일 선택")
input_mode = st.radio("입력 방식", options=["내 컴퓨터에서 업로드", "샘플 데이터 생성 (테스트용)", "서버 폴더에서 선택"], horizontal=True)

selected_source_path = None
selected_source_name = None
selected_source_token = None
uploaded_file = None

if input_mode == "내 컴퓨터에서 업로드":
    uploaded_file = st.file_uploader("NIfTI 파일 업로드", type=None)
    if uploaded_file:
        selected_source_name = uploaded_file.name
        selected_source_token = make_source_token_from_upload(uploaded_file)
elif input_mode == "샘플 데이터 생성 (테스트용)":
    if st.button("가상 3D 샘플 데이터 생성"):
        st.session_state.sample_path = generate_synthetic_nifti()
        st.success("샘플 데이터 생성 완료!")
    if "sample_path" in st.session_state:
        selected_source_path = st.session_state.sample_path
        selected_source_name = "synthetic_sample.nii.gz"
        selected_source_token = "synthetic::sample"
else:
    folder_path = st.text_input("폴더 경로", value=".")
    nifti_files = list_nifti_files(folder_path)
    if nifti_files:
        selected_source_path = st.selectbox("파일 선택", options=nifti_files, format_func=lambda x: Path(x).name)
        selected_source_name = Path(selected_source_path).name
        selected_source_token = make_source_token_from_path(selected_source_path)

if "source_token" not in st.session_state or st.session_state.source_token != selected_source_token:
    reset_inference_state()
    st.session_state.source_token = selected_source_token

# ============================================================
# 6. 추론 실행
# ============================================================
if st.button("추론 실행", type="primary"):
    path = save_uploaded_file_to_temp(uploaded_file) if uploaded_file else selected_source_path
    if path:
        with st.spinner("분석 중..."):
            data = preprocess_case(path)
            result = run_inference(data, model, device_name, (roi_x, roi_y, roi_z))
            st.session_state.result = result
            st.session_state.raw_shape = nib.load(path).shape
            st.session_state.inference_done = True
            st.success("분석 완료!")

# ============================================================
# 7. 결과 표시
# ============================================================
if st.session_state.inference_done and st.session_state.result:
    res = st.session_state.result
    raw_pred_np = res["raw_pred_np"]
    st.write(f"추론 시간: {res['elapsed']:.2f}s | Shape: {st.session_state.raw_shape}")
    
    axis = st.selectbox("축", [0, 1, 2], format_func=lambda x: ["Sagittal", "Coronal", "Axial"][x], index=2)
    idx = st.slider("Slice", 0, res["image_np"].shape[axis]-1, get_best_slice(raw_pred_np, axis))
    st.pyplot(make_overlay_figure(extract_slice(res["image_np"], axis, idx), extract_slice(raw_pred_np, axis, idx)))
    
    st.markdown("---")
    st.subheader("🧊 3D 입체 시각화")
    fig_3d = visualize_3d_mask(raw_pred_np)
    if fig_3d: st.plotly_chart(fig_3d, use_container_width=True)
    else: st.warning("표시할 영역이 없습니다.")
