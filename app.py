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
    recursive=True면 하위 폴더까지 포함
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"폴더가 존재하지 않습니다: {folder_path}")

    if not folder.is_dir():
        raise NotADirectoryError(f"폴더 경로가 아닙니다: {folder_path}")

    if recursive:
        candidates = folder.rglob("*")
    else:
        candidates = folder.iterdir()

    files = []
    for p in candidates:
        if p.is_file() and is_nifti_file_name(p.name):
            files.append(str(p.resolve()))

    files.sort()
    return files


def make_source_token_from_path(file_path: str) -> str:
    """파일 경로 기반 token 생성"""
    p = Path(file_path)
    stat = p.stat()
    return f"path::{str(p.resolve())}::{stat.st_size}::{stat.st_mtime}"


def make_source_token_from_upload(uploaded_file) -> str:
    """업로드 파일 기반 token 생성"""
    return f"upload::{uploaded_file.name}::{uploaded_file.size}"


def generate_synthetic_nifti():
    """테스트용 가상 3D NIfTI 파일 생성 (구 형태)"""
    size = (96, 96, 96)
    data = np.zeros(size, dtype=np.float32)
    
    # 중앙에 구 생성 (장기 모사)
    z, y, x = np.ogrid[:size[0], :size[1], :size[2]]
    center = (48, 48, 48)
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    data[dist_from_center <= 25] = 100.0  # 내부 값
    
    # 노이즈 및 배경 추가 (실제 CT 느낌)
    data += np.random.normal(-50, 20, size)
    
    affine = np.eye(4)
    image = nib.Nifti1Image(data, affine)
    
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        nib.save(image, tmp.name)
        return tmp.name


def get_available_device_options():
    """
    사용 가능한 디바이스 목록 반환
    """
    options = []
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        options.append(("mps", "GPU (Apple Silicon / MPS)"))
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            options.append((f"cuda:{i}", f"GPU (CUDA:{i} / {gpu_name})"))
    options.append(("cpu", "CPU"))
    return options


def get_device_debug_text():
    """디바이스 정보 텍스트"""
    lines = [f"torch.__version__ = {torch.__version__}", f"torch.cuda.is_available() = {torch.cuda.is_available()}"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            lines.append(f"cuda:{i} = {torch.cuda.get_device_name(i)}")
    return "\n".join(lines)


def build_model() -> UNet:
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )
    return model


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


def keep_largest_connected_component_np(pred_np: np.ndarray, target_label: int = 1) -> Tuple[np.ndarray, str]:
    mask = (pred_np == target_label).astype(np.uint8)
    if mask.sum() == 0: return pred_np.copy(), "양성 영역이 없습니다."
    try:
        from scipy import ndimage as ndi
        structure = ndi.generate_binary_structure(rank=3, connectivity=1)
        labeled, num_components = ndi.label(mask, structure=structure)
        if num_components == 0: return pred_np.copy(), "컴포넌트를 찾지 못했습니다."
        component_sizes = np.bincount(labeled.ravel())
        component_sizes[0] = 0
        largest_label = int(np.argmax(component_sizes))
        largest_mask = (labeled == largest_label)
        processed = np.zeros_like(pred_np, dtype=np.uint8)
        processed[largest_mask] = target_label
        return processed, f"가장 큰 컴포넌트만 유지 ({num_components}개 중 1개)"
    except Exception as e:
        return pred_np.copy(), f"후처리 실패: {e}"


def visualize_3d_mask(mask_np):
    """Plotly를 이용한 3D Segmentation 마스크 시각화"""
    # 성능을 위해 듬성듬성 샘플링
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


def extract_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0: return volume[index, :, :]
    elif axis == 1: return volume[:, index, :]
    else: return volume[:, :, index]


def get_best_slice(mask_3d: np.ndarray, axis: int) -> int:
    if mask_3d.sum() == 0: return mask_3d.shape[axis] // 2
    slice_sums = [extract_slice(mask_3d, axis, i).sum() for i in range(mask_3d.shape[axis])]
    return int(np.argmax(slice_sums))


def make_overlay_figure(image_slice, pred_slice):
    def norm(img):
        mn, mx = img.min(), img.max()
        return (img - mn) / (mx - mn) if mx - mn > 1e-8 else np.zeros_like(img)
    
    img_disp = norm(image_slice)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_disp, cmap="gray"); axes[0].set_title("Image"); axes[0].axis("off")
    axes[1].imshow(pred_slice, cmap="gray"); axes[1].set_title("Prediction"); axes[1].axis("off")
    axes[2].imshow(img_disp, cmap="gray"); axes[2].imshow(pred_slice, cmap="Reds", alpha=0.35)
    axes[2].set_title("Overlay"); axes[2].axis("off")
    plt.tight_layout()
    return fig


def make_nifti_bytes(pred_np, affine):
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        img = nib.Nifti1Image(pred_np.astype(np.uint8), affine=affine)
        nib.save(img, tmp.name)
        with open(tmp.name, "rb") as f: data = f.read()
    if os.path.exists(tmp.name): os.remove(tmp.name)
    return data


# ============================================================
# 3. 사이드바 및 모델 로드
# ============================================================
with st.sidebar:
    st.header("실행 설정")
    model_path = st.text_input("모델 파일 경로 (.pth)", value="./best_metric_model.pth")
    device_items = get_available_device_options()
    device_name = st.selectbox("디바이스 선택", options=[v for v, _ in device_items], format_func=lambda x: dict(device_items)[x])
    roi_x = st.number_input("ROI X", 32, 256, 160, 16)
    roi_y = st.number_input("ROI Y", 32, 256, 160, 16)
    roi_z = st.number_input("ROI Z", 32, 256, 160, 16)
    keep_largest_cc = st.checkbox("가장 큰 컴포넌트만 유지", False)

try:
    model = load_model_cached(model_path, device_name)
    st.sidebar.success("모델 로드 완료")
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
    uploaded_file = st.file_uploader("NIfTI 파일 업로드 (.nii / .nii.gz)", type=None)
    if uploaded_file and is_nifti_file_name(uploaded_file.name):
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

if st.session_state.source_token != selected_source_token:
    reset_inference_state()
    st.session_state.source_token = selected_source_token

# ============================================================
# 6. 추론 실행
# ============================================================
if st.button("추론 실행", type="primary"):
    temp_path = None
    try:
        path = save_uploaded_file_to_temp(uploaded_file) if uploaded_file else selected_source_path
        if path:
            with st.spinner("추론 중..."):
                data = preprocess_case(path)
                result = run_inference(data, model, device_name, (roi_x, roi_y, roi_z))
                st.session_state.result = result
                st.session_state.raw_shape = nib.load(path).shape
                st.session_state.inference_done = True
                st.success("완료!")
    except Exception as e: st.exception(e)

# ============================================================
# 7. 결과 표시
# ============================================================
if st.session_state.inference_done and st.session_state.result:
    res = st.session_state.result
    raw_pred_np = res["raw_pred_np"]
    
    st.write(f"추론 시간: {res['elapsed']:.2f}s")
    
    # 2D 시각화
    axis = st.selectbox("축 선택", [0, 1, 2], format_func=lambda x: ["Sagittal", "Coronal", "Axial"][x], index=2)
    idx = st.slider("Slice", 0, res["image_np"].shape[axis]-1, get_best_slice(raw_pred_np, axis))
    st.pyplot(make_overlay_figure(extract_slice(res["image_np"], axis, idx), extract_slice(raw_pred_np, axis, idx)))
    
    # 3D 시각화
    st.markdown("---")
    st.subheader("🧊 3D 입체 시각화")
    fig_3d = visualize_3d_mask(raw_pred_np)
    if fig_3d: st.plotly_chart(fig_3d, use_container_width=True)
    else: st.warning("표시할 영역이 없습니다.")
    
    st.download_button("결과 다운로드 (.nii.gz)", make_nifti_bytes(raw_pred_np, res["affine"]), "result.nii.gz")
