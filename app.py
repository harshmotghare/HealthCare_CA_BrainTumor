"""Multimodal Brain Tumor Segmentation Using MRI Data and Explainable AI.

Main Streamlit Application Dashboard.
"""

import io
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from data.preprocessing import prepare_input_tensor
from data.validation import (
    REQUIRED_MODALITIES,
    ValidationResult,
    extract_modalities_from_zip,
    load_nifti_from_source,
    validate_modalities,
)
from evaluation.metrics import decode_ground_truth_subregions, evaluate_prediction
from explainability.modality_ablation import run_modality_ablation
from explainability.saliency import compute_gradient_saliency
from inference.export import export_mask_to_bytes
from inference.predictor import SegmentationResult, run_segmentation
from models.pretrained_model import load_pretrained_model
from ui.styles import (
    COLOR_PRIMARY_NAVY,
    COLOR_TEAL,
    SUBREGION_COLORS,
    get_custom_css,
)

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Segmentation & XAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading pretrained SegResNet bundle...")
def get_model():
    """Load and cache the pretrained SegResNet model."""
    try:
        model, device, meta = load_pretrained_model()
        return model, device, meta, None
    except Exception as e:
        return None, None, None, str(e)


def init_session_state():
    """Initialize persistent session state variables."""
    defaults = {
        "modalities_data": None,
        "validation_result": None,
        "segmentation_result": None,
        "reference_affine": None,
        "reference_header": None,
        "ground_truth_map": None,
        "ground_truth_evaluated": False,
        "evaluation_report": None,
        "saliency_result": None,
        "ablation_report": None,
        "active_patient_id": "Unassigned",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()
model, device, model_meta, model_error = get_model()


# -----------------------------------------------------------------------------
# HELPER RENDERING FUNCTIONS
# -----------------------------------------------------------------------------

def get_2d_slice(volume: np.ndarray, plane: str, slice_idx: int) -> np.ndarray:
    """Extract a 2D slice from a 3D volume along the specified anatomical plane."""
    if plane == "Axial":
        # (Z-axis) slice along index 2
        return volume[:, :, slice_idx].T
    elif plane == "Coronal":
        # (Y-axis) slice along index 1
        return volume[:, slice_idx, :].T
    else:  # Sagittal
        # (X-axis) slice along index 0
        return volume[slice_idx, :, :].T


def create_slice_overlay(
    mri_slice: np.ndarray,
    wt_slice: Optional[np.ndarray] = None,
    tc_slice: Optional[np.ndarray] = None,
    et_slice: Optional[np.ndarray] = None,
    saliency_slice: Optional[np.ndarray] = None,
    show_wt: bool = True,
    show_tc: bool = True,
    show_et: bool = True,
    show_saliency: bool = False,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend MRI background slice with colored subregion masks or saliency map."""
    # Normalize MRI slice to [0, 1] for RGB display
    m_min, m_max = np.min(mri_slice), np.max(mri_slice)
    if m_max > m_min:
        mri_norm = (mri_slice - m_min) / (m_max - m_min)
    else:
        mri_norm = np.zeros_like(mri_slice, dtype=np.float32)

    # 3-channel grayscale image
    rgb = np.stack([mri_norm, mri_norm, mri_norm], axis=-1)

    # Apply Saliency Heatmap if requested
    if show_saliency and saliency_slice is not None and np.max(saliency_slice) > 0:
        cmap = plt.get_cmap("inferno")
        saliency_colored = cmap(saliency_slice)[:, :, :3]  # drop alpha
        mask_active = saliency_slice > 0.05
        for c in range(3):
            rgb[:, :, c] = np.where(
                mask_active,
                (1 - alpha) * rgb[:, :, c] + alpha * saliency_colored[:, :, c],
                rgb[:, :, c],
            )
        return np.clip(rgb, 0.0, 1.0)

    # Color overlays for tumor subregions:
    # Blend in hierarchical order: WT (yellow/amber) -> TC (red) -> ET (cyan)
    if show_wt and wt_slice is not None:
        c = SUBREGION_COLORS["WT"]["rgb"]
        active = wt_slice > 0
        for i in range(3):
            rgb[:, :, i] = np.where(active, (1 - alpha) * rgb[:, :, i] + alpha * c[i], rgb[:, :, i])

    if show_tc and tc_slice is not None:
        c = SUBREGION_COLORS["TC"]["rgb"]
        active = tc_slice > 0
        for i in range(3):
            rgb[:, :, i] = np.where(active, (1 - alpha) * rgb[:, :, i] + alpha * c[i], rgb[:, :, i])

    if show_et and et_slice is not None:
        c = SUBREGION_COLORS["ET"]["rgb"]
        active = et_slice > 0
        for i in range(3):
            rgb[:, :, i] = np.where(active, (1 - alpha) * rgb[:, :, i] + alpha * c[i], rgb[:, :, i])

    return np.clip(rgb, 0.0, 1.0)


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & SYSTEM STATUS
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🧠 NeuroSeg XAI")
    st.caption("Multimodal MRI Brain Tumor Segmentation & Explainable AI")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📤 MRI Upload & Validation",
            "🧠 Segmentation Results",
            "🔬 Explainable AI (XAI)",
            "📊 Evaluation & Metrics",
            "📖 Methodology",
            "ℹ️ About & Honesty",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### System Hardware & Model")
    if model_error:
        st.error(f"❌ Model Checkpoint Error:\n{model_error}")
    else:
        st.success(f"✅ SegResNet Loaded ({device})")
        st.caption(f"Weights: `{Path(model_meta['checkpoint_path']).name}`")
        st.caption(f"Parameters: {model_meta['parameters']:,}")

    st.markdown("---")
    st.markdown("#### Loaded Patient Data")
    if st.session_state["modalities_data"] is not None:
        st.info(f"📁 Patient: **{st.session_state['active_patient_id']}**")
        val_res = st.session_state["validation_result"]
        if val_res and val_res.common_shape:
            st.caption(f"Shape: {val_res.common_shape}")
            st.caption(f"Spacing: {val_res.common_spacing} mm")
    else:
        st.write("No MRI volume loaded yet.")

    st.markdown("---")
    st.caption("Academic Project • Research & Educational Use Only")


# =============================================================================
# 1. HOME PAGE
# =============================================================================

if page == "🏠 Home":
    st.markdown(
        """
        <div class="hero-banner">
            <h1>Multimodal Brain Tumor Segmentation Using MRI Data & Explainable AI</h1>
            <p>A deep-learning clinical decision-support framework powered by MONAI SegResNet, multi-sequence MRI fusion, and gradient attribution explainability.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Abstract & Clinical Motivation")
        st.markdown(
            """
            Gliomas are the most prevalent and aggressive primary malignant brain neoplasms. Accurately delineating 
            tumor subregions—including the **Whole Tumor (WT)**, the resectable **Tumor Core (TC)**, and active **Enhancing Tumor (ET)**—is 
            paramount for surgical resection planning, radiation oncology, and treatment monitoring.
            
            Because distinct pathological characteristics present across different imaging sequences, single-modality MRI is insufficient:
            - **T1-weighted (T1):** Provides sharp structural anatomy and healthy tissue contrast.
            - **Contrast-Enhanced T1 (T1c / T1Gd):** Emphasizes blood-brain barrier breakdown and active tumor margins.
            - **T2-weighted (T2):** Visualizes fluid accumulation and general lesion mass.
            - **FLAIR:** Suppresses cerebrospinal fluid (CSF) signals to clearly expose **peritumoral edema**.
            
            This application implements an end-to-end medical pipeline that couples the **MONAI SegResNet** backbone with 
            rigorous spatial geometry verification, multi-planar slice inspection, and **Explainable AI (XAI)** to illuminate 
            how multi-sequence inputs guide pathological segmentation.
            """
        )

        st.markdown("### Project Workflow")
        st.markdown(
            """
            ```
            [ 4 MRI Modalities: T1, T1c, T2, FLAIR ]
                          │
                          ▼
            [ NiBabel 3D Validation & Affine Alignment ]
                          │
                          ▼
            [ Nonzero Channel-wise Intensity Normalization ]
                          │
                          ▼
            [ MONAI Pretrained SegResNet Inference (Sliding Window) ]
                          │
                          ▼
            [ Sigmoid Activation & Subregion Multi-class Decoding ]
              ├── Whole Tumor (WT) = Edema + Core + Enhancing
              ├── Tumor Core (TC)  = Necrosis + Enhancing
              └── Enhancing Tumor  = Active Contrast Margin
                          │
                          ├─────────────────────────┬─────────────────────────┐
                          ▼                         ▼                         ▼
            [ Multi-Planar Viewer ]      [ Explainable AI (XAI) ]   [ Ground-Truth Metrics ]
            - Axial / Coronal / Sagittal - Gradient Saliency Map     - Dice & IoU Scores
            - Real-time Alpha Blending   - Modality Ablation Test    - HD95 Surface Distance
            - NIfTI 3D Mask Export       - Clinical Attribution      - Sensitivity & Specificity
            ```
            """
        )

    with col2:
        st.markdown("### Model & System Status")
        st.markdown(
            """
            <div class="clinical-card">
                <h4>Backbone Architecture</h4>
                <p><strong>MONAI SegResNet (Myronenko 2018)</strong></p>
                <span class="badge badge-teal">Pretrained Weights Verified</span>
                <span class="badge badge-navy">BraTS 2018 Benchmark</span>
                <hr style="margin: 12px 0;">
                <p><strong>Benchmark Validation Dice:</strong></p>
                <ul>
                    <li>Whole Tumor (WT): <strong>0.9026</strong></li>
                    <li>Tumor Core (TC): <strong>0.8559</strong></li>
                    <li>Enhancing Tumor (ET): <strong>0.7905</strong></li>
                    <li>Mean Overall: <strong>0.8518</strong></li>
                </ul>
                <hr style="margin: 12px 0;">
                <p><strong>Input Shape:</strong> 4 Channels × 1.0mm³ Isotropic</p>
                <p><strong>Inference Mode:</strong> 3D Sliding Window with Gaussian Weighting</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="clinical-card">
                <h4>Quick Start Options</h4>
                <p>1. Go to <strong>MRI Upload & Validation</strong>.</p>
                <p>2. Click <strong>"Load Sample Patient (BraTS)"</strong> for an instant demo without manual uploads.</p>
                <p>3. Or upload your own 4 NIfTI files or ZIP bundle.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# 2. MRI UPLOAD & VALIDATION PAGE
# =============================================================================

elif page == "📤 MRI Upload & Validation":
    st.markdown("## 📤 Multimodal MRI Upload & Spatial Validation")
    st.write(
        "Upload the 4 core clinical MRI sequences in `.nii` or `.nii.gz` format, "
        "or upload a single `.zip` containing all four sequences."
    )

    sample_col1, sample_col2 = st.columns([1, 1])
    with sample_col1:
        st.markdown("#### Option A: Quick Demonstration Sample")
        st.write("Load a pre-packaged 3D multimodal brain volume with ground truth for immediate verification:")
        if st.button("🚀 Load Pre-packaged Sample Patient (3D BraTS Phantom)", use_container_width=True):
            sample_dir = Path("data_local/sample_patient")
            t1_p = sample_dir / "T1.nii.gz"
            t1c_p = sample_dir / "T1CE.nii.gz"
            t2_p = sample_dir / "T2.nii.gz"
            flair_p = sample_dir / "FLAIR.nii.gz"
            truth_p = sample_dir / "truth.nii.gz"

            if all(p.is_file() for p in [t1_p, t1c_p, t2_p, flair_p]):
                mod_inputs = {
                    "t1ce": str(t1c_p),
                    "t1": str(t1_p),
                    "t2": str(t2_p),
                    "flair": str(flair_p),
                }
                v_res = validate_modalities(mod_inputs)
                if v_res.is_valid:
                    # Load numpy arrays
                    loaded_arrs = {
                        k: load_nifti_from_source(mod_inputs[k]).get_fdata() for k in REQUIRED_MODALITIES
                    }
                    st.session_state["modalities_data"] = loaded_arrs
                    st.session_state["validation_result"] = v_res
                    st.session_state["reference_affine"] = v_res.reference_affine
                    st.session_state["reference_header"] = v_res.reference_header
                    st.session_state["active_patient_id"] = "Sample_Patient_BraTS_01"
                    st.session_state["segmentation_result"] = None

                    # Load truth if present
                    if truth_p.is_file():
                        t_img = nib.load(str(truth_p))
                        st.session_state["ground_truth_map"] = t_img.get_fdata().astype(np.uint8)

                    st.success("✅ Sample patient loaded and validated successfully!")
            else:
                st.error("Sample patient files were not found on disk at `data_local/sample_patient`.")

    with sample_col2:
        st.markdown("#### Option B: Upload ZIP Archive")
        zip_file = st.file_uploader("Upload .zip archive with T1, T1c, T2, FLAIR", type=["zip"])
        if zip_file is not None:
            extracted = extract_modalities_from_zip(zip_file)
            if len(extracted) == 4:
                mod_inputs = {k: extracted[k][1] for k in REQUIRED_MODALITIES}
                v_res = validate_modalities(mod_inputs)
                if v_res.is_valid:
                    loaded_arrs = {
                        k: load_nifti_from_source(extracted[k][1], filename=extracted[k][0]).get_fdata()
                        for k in REQUIRED_MODALITIES
                    }
                    st.session_state["modalities_data"] = loaded_arrs
                    st.session_state["validation_result"] = v_res
                    st.session_state["reference_affine"] = v_res.reference_affine
                    st.session_state["reference_header"] = v_res.reference_header
                    st.session_state["active_patient_id"] = zip_file.name.replace(".zip", "")
                    st.session_state["segmentation_result"] = None
                    st.success("✅ ZIP extracted and 4 modalities validated!")
            else:
                st.error(f"ZIP archive must contain all 4 modalities. Detected: {list(extracted.keys())}")

    st.markdown("---")
    st.markdown("#### Option C: Upload Individual Sequence Files")
    up_c1, up_c2, up_c3, up_c4 = st.columns(4)

    with up_c1:
        st.markdown("**1. Contrast T1 (T1c / T1Gd)**")
        f_t1ce = st.file_uploader("T1-CE scan", type=["nii", "nii.gz"], key="up_t1ce")
    with up_c2:
        st.markdown("**2. Native T1**")
        f_t1 = st.file_uploader("T1 scan", type=["nii", "nii.gz"], key="up_t1")
    with up_c3:
        st.markdown("**3. T2-Weighted**")
        f_t2 = st.file_uploader("T2 scan", type=["nii", "nii.gz"], key="up_t2")
    with up_c4:
        st.markdown("**4. T2-FLAIR**")
        f_flair = st.file_uploader("FLAIR scan", type=["nii", "nii.gz"], key="up_flair")

    if f_t1ce and f_t1 and f_t2 and f_flair:
        mod_inputs = {"t1ce": f_t1ce, "t1": f_t1, "t2": f_t2, "flair": f_flair}
        v_res = validate_modalities(mod_inputs)
        if v_res.is_valid:
            loaded_arrs = {k: load_nifti_from_source(mod_inputs[k]).get_fdata() for k in REQUIRED_MODALITIES}
            st.session_state["modalities_data"] = loaded_arrs
            st.session_state["validation_result"] = v_res
            st.session_state["reference_affine"] = v_res.reference_affine
            st.session_state["reference_header"] = v_res.reference_header
            st.session_state["active_patient_id"] = Path(f_t1ce.name).stem.split("_")[0]
            st.session_state["segmentation_result"] = None
            st.success("✅ All 4 individual sequences successfully validated!")

    # Display Validation Status and Metadata
    val_res: Optional[ValidationResult] = st.session_state["validation_result"]
    if val_res is not None:
        st.markdown("---")
        st.markdown("### 📋 Validation Report & Spatial Summary")

        if not val_res.is_valid:
            st.error("Validation failed with errors:")
            for err in val_res.errors:
                st.write(f"- ❌ {err}")
        else:
            if val_res.warnings:
                for w in val_res.warnings:
                    st.warning(f"⚠️ {w}")

            vcols = st.columns(4)
            for idx, (k, meta) in enumerate(val_res.modalities.items()):
                with vcols[idx]:
                    st.markdown(
                        f"""
                        <div class="clinical-card">
                            <h4>{meta.name}</h4>
                            <p><strong>File:</strong> <code>{meta.filename}</code></p>
                            <p><strong>Shape:</strong> {meta.shape}</p>
                            <p><strong>Spacing:</strong> {meta.voxel_spacing[0]:.2f} × {meta.voxel_spacing[1]:.2f} × {meta.voxel_spacing[2]:.2f} mm</p>
                            <p><strong>Range:</strong> [{meta.min_intensity:.1f}, {meta.max_intensity:.1f}]</p>
                            <p><strong>Mean:</strong> {meta.mean_intensity:.1f}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Preprocessing summary card
            st.markdown(
                """
                <div class="clinical-card">
                    <h4>Preprocessing Pipeline Summary</h4>
                    <ul>
                        <li><strong>Input Channel Mapping:</strong> <code>Channel 0: T1c</code>, <code>Channel 1: T1</code>, <code>Channel 2: T2</code>, <code>Channel 3: FLAIR</code></li>
                        <li><strong>Intensity Normalization:</strong> Nonzero voxel channel-wise standardization (zero mean, unit variance within brain parenchyma).</li>
                        <li><strong>Symmetric Spatial Padding:</strong> Divisible by 16 for encoder downsampling; automatically inverted post-inference.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # RUN SEGMENTATION BUTTON
            st.markdown("---")
            run_col1, run_col2 = st.columns([2, 1])
            with run_col1:
                st.markdown("#### Ready to Segment")
                st.write("Execute 3D deep learning segmentation using the pretrained MONAI SegResNet.")
            with run_col2:
                if st.button("🚀 Run Segmentation", type="primary", use_container_width=True):
                    with st.spinner("Executing 3D SegResNet segmentation..."):
                        prog_bar = st.progress(0.1)
                        status_text = st.empty()

                        def update_prog(val, msg):
                            prog_bar.progress(val)
                            status_text.text(msg)

                        # Preprocess input tensor
                        input_tensor, pad_amounts, orig_shape = prepare_input_tensor(
                            modality_arrays=st.session_state["modalities_data"],
                            device=device,
                        )

                        seg_res = run_segmentation(
                            model=model,
                            input_tensor=input_tensor,
                            pad_amounts=pad_amounts,
                            voxel_spacing=val_res.common_spacing or (1.0, 1.0, 1.0),
                            progress_callback=update_prog,
                        )

                        st.session_state["segmentation_result"] = seg_res
                        prog_bar.empty()
                        status_text.empty()
                        st.success(f"🎉 Segmentation completed in {seg_res.inference_time_sec:.2f} seconds!")
                        st.info("Navigate to the **'Segmentation Results'** tab to inspect the 3D overlays!")


# =============================================================================
# 3. SEGMENTATION RESULTS PAGE
# =============================================================================

elif page == "🧠 Segmentation Results":
    st.markdown("## 🧠 Segmentation Results & Multi-Planar Viewer")

    seg_res: Optional[SegmentationResult] = st.session_state.get("segmentation_result")
    mod_data = st.session_state.get("modalities_data")

    if seg_res is None or mod_data is None:
        st.warning("No segmentation results found. Please upload MRI scans and run segmentation first in **'MRI Upload & Validation'**.")
    else:
        # Volumetric Metric KPI Cards
        st.markdown("### 📊 Tumor Subregion Volumetrics")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:
            wt_m = seg_res.metrics["WT"]
            st.markdown(
                f"""
                <div class="metric-card" style="border-left-color: {SUBREGION_COLORS['WT']['hex']};">
                    <div class="metric-title">Whole Tumor (WT)</div>
                    <div class="metric-value">{wt_m.volume_cm3:.2f} cm³</div>
                    <div class="metric-sub">{wt_m.voxel_count:,} voxels • Max Prob {wt_m.max_probability:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with kpi2:
            tc_m = seg_res.metrics["TC"]
            st.markdown(
                f"""
                <div class="metric-card" style="border-left-color: {SUBREGION_COLORS['TC']['hex']};">
                    <div class="metric-title">Tumor Core (TC)</div>
                    <div class="metric-value">{tc_m.volume_cm3:.2f} cm³</div>
                    <div class="metric-sub">{tc_m.voxel_count:,} voxels • Mean Prob {tc_m.mean_probability:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with kpi3:
            et_m = seg_res.metrics["ET"]
            st.markdown(
                f"""
                <div class="metric-card" style="border-left-color: {SUBREGION_COLORS['ET']['hex']};">
                    <div class="metric-title">Enhancing Tumor (ET)</div>
                    <div class="metric-value">{et_m.volume_cm3:.2f} cm³</div>
                    <div class="metric-sub">{et_m.voxel_count:,} voxels • Mean Prob {et_m.mean_probability:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with kpi4:
            st.markdown(
                f"""
                <div class="metric-card" style="border-left-color: {COLOR_PRIMARY_NAVY};">
                    <div class="metric-title">Inference Speed</div>
                    <div class="metric-value">{seg_res.inference_time_sec:.2f}s</div>
                    <div class="metric-sub">Est. Brain Vol: {seg_res.brain_volume_cm3:.1f} cm³</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Visualizer Controls
        ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([1, 1, 2])

        with ctrl_c1:
            plane = st.selectbox("Anatomical View Plane", ["Axial", "Coronal", "Sagittal"])
            bg_modality = st.selectbox(
                "Background MRI Modality",
                ["T1-CE (t1ce)", "Native T1 (t1)", "T2-Weighted (t2)", "T2-FLAIR (flair)"],
                index=0,
            )
            bg_key = "t1ce" if "t1ce" in bg_modality else ("t1" if "t1" in bg_modality else ("t2" if "t2" in bg_modality else "flair"))

        with ctrl_c2:
            st.markdown("**Subregion Visibility**")
            show_wt = st.checkbox("Whole Tumor (WT)", value=True)
            show_tc = st.checkbox("Tumor Core (TC)", value=True)
            show_et = st.checkbox("Enhancing Tumor (ET)", value=True)
            alpha = st.slider("Overlay Opacity (Alpha)", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

        # Determine slice count for selected plane
        bg_vol = mod_data[bg_key]
        if plane == "Axial":
            max_slices = bg_vol.shape[2] - 1
            default_slice = max_slices // 2
        elif plane == "Coronal":
            max_slices = bg_vol.shape[1] - 1
            default_slice = max_slices // 2
        else:  # Sagittal
            max_slices = bg_vol.shape[0] - 1
            default_slice = max_slices // 2

        with ctrl_c3:
            st.markdown("**Slice Navigation**")
            slice_idx = st.slider(f"Select {plane} Slice Index", 0, max_slices, default_slice)
            st.caption(f"Currently inspecting slice {slice_idx} of {max_slices} in {plane} projection.")

        # Extract 2D slices
        mri_2d = get_2d_slice(bg_vol, plane, slice_idx)
        wt_2d = get_2d_slice(seg_res.binary_masks[1], plane, slice_idx)
        tc_2d = get_2d_slice(seg_res.binary_masks[0], plane, slice_idx)
        et_2d = get_2d_slice(seg_res.binary_masks[2], plane, slice_idx)

        # Render Slice Overlay
        overlay_rgb = create_slice_overlay(
            mri_slice=mri_2d,
            wt_slice=wt_2d,
            tc_slice=tc_2d,
            et_slice=et_2d,
            show_wt=show_wt,
            show_tc=show_tc,
            show_et=show_et,
            alpha=alpha,
        )

        view_c1, view_c2 = st.columns([3, 1])

        with view_c1:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(overlay_rgb, origin="lower")
            ax.axis("off")
            ax.set_title(f"{plane} Slice {slice_idx} • {bg_modality}", color=COLOR_PRIMARY_NAVY, fontsize=12, pad=10)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with view_c2:
            st.markdown("#### Color Legend")
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <div style="width: 20px; height: 20px; background-color: {SUBREGION_COLORS['WT']['hex']}; border-radius: 4px; margin-right: 10px;"></div>
                    <div><strong>Whole Tumor (WT)</strong><br><small>Edema + Core + Enhancing</small></div>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <div style="width: 20px; height: 20px; background-color: {SUBREGION_COLORS['TC']['hex']}; border-radius: 4px; margin-right: 10px;"></div>
                    <div><strong>Tumor Core (TC)</strong><br><small>Necrosis + Enhancing</small></div>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <div style="width: 20px; height: 20px; background-color: {SUBREGION_COLORS['ET']['hex']}; border-radius: 4px; margin-right: 10px;"></div>
                    <div><strong>Enhancing Tumor (ET)</strong><br><small>Active contrast margin</small></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("---")
            st.markdown("#### 💾 Export NIfTI Mask")
            st.write("Download the predicted 3D segmentation mask with original patient spatial affine:")

            # Prepare downloadable bytes
            ref_affine = st.session_state.get("reference_affine")
            if ref_affine is None:
                ref_affine = np.eye(4)
            ref_header = st.session_state.get("reference_header")

            # Option 1: Multi-class BraTS discrete map
            nii_bytes_multi = export_mask_to_bytes(
                mask_data=seg_res.multiclass_map,
                reference_affine=ref_affine,
                reference_header=ref_header,
            )
            st.download_button(
                label="📥 Download BraTS Label Map (.nii.gz)",
                data=nii_bytes_multi,
                file_name=f"{st.session_state['active_patient_id']}_seg_brats.nii.gz",
                mime="application/gzip",
                use_container_width=True,
            )

            # Option 2: Whole Tumor binary mask
            nii_bytes_wt = export_mask_to_bytes(
                mask_data=seg_res.binary_masks[1].astype(np.uint8),
                reference_affine=ref_affine,
                reference_header=ref_header,
            )
            st.download_button(
                label="📥 Download Whole Tumor Mask (.nii.gz)",
                data=nii_bytes_wt,
                file_name=f"{st.session_state['active_patient_id']}_wt_mask.nii.gz",
                mime="application/gzip",
                use_container_width=True,
            )


# =============================================================================
# 4. EXPLAINABLE AI (XAI) PAGE
# =============================================================================

elif page == "🔬 Explainable AI (XAI)":
    st.markdown("## 🔬 Explainable AI (XAI) & Attribution Analysis")
    st.write(
        "Inspect genuine gradient-based attributions and sequence ablation to understand "
        "how the deep learning network integrates multimodal MRI signals."
    )

    seg_res: Optional[SegmentationResult] = st.session_state.get("segmentation_result")
    mod_data = st.session_state.get("modalities_data")

    if seg_res is None or mod_data is None:
        st.warning("Please run segmentation first before computing Explainable AI attributions.")
    else:
        xai_tabs = st.tabs(["🔥 Gradient Saliency Map", "📉 Modality Ablation Analysis", "📖 Scientific Interpretation"])

        # TAB 1: GRADIENT SALIENCY
        with xai_tabs[0]:
            st.markdown("### Genuine Backpropagation Gradient Saliency")
            st.write(
                "Computes exact gradients of the network output with respect to the input multimodal volume: "
                r"$\nabla_X S = \frac{\partial S_c}{\partial X}$."
            )

            xai_c1, xai_c2 = st.columns([1, 2])
            with xai_c1:
                target_sub = st.selectbox(
                    "Target Pathology Subregion",
                    ["Enhancing Tumor (ET)", "Tumor Core (TC)", "Whole Tumor (WT)"],
                    index=0,
                )
                sub_code = "ET" if "ET" in target_sub else ("TC" if "TC" in target_sub else "WT")

                patch_size_opt = st.selectbox("Analysis Patch Size", ["64 x 64 x 64 voxels", "48 x 48 x 48 voxels"], index=0)
                pz = 64 if "64" in patch_size_opt else 48

                if st.button("⚡ Compute Gradient Saliency", type="primary", use_container_width=True):
                    with st.spinner(f"Computing backpropagation gradients for {target_sub}..."):
                        input_tensor, pad_amounts, _ = prepare_input_tensor(mod_data, device=device)
                        # Reference mask for centering
                        ref_mask = seg_res.binary_masks[1]  # WT mask

                        saliency_res = compute_gradient_saliency(
                            model=model,
                            input_tensor=input_tensor,
                            target_subregion=sub_code,
                            reference_mask=ref_mask,
                            patch_size=(pz, pz, pz),
                            device=device,
                        )
                        st.session_state["saliency_result"] = saliency_res
                        st.success("✅ Gradient Saliency successfully calculated!")

            saliency_res: Optional[SaliencyResult] = st.session_state.get("saliency_result")

            with xai_c2:
                if saliency_res is not None:
                    # Bar chart of modality attribution
                    df_attrib = {
                        "Modality": list(saliency_res.modality_percentages.keys()),
                        "Attribution (%)": list(saliency_res.modality_percentages.values()),
                    }
                    fig_bar = px.bar(
                        df_attrib,
                        x="Modality",
                        y="Attribution (%)",
                        color="Modality",
                        title=f"Sequence Attribution for {saliency_res.target_subregion}",
                        text_auto=".1f",
                        color_discrete_sequence=["#06b6d4", "#0f2b48", "#10b981", "#f59e0b"],
                    )
                    fig_bar.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_bar, use_container_width=True)

            if saliency_res is not None:
                st.markdown("---")
                st.markdown("#### Saliency Heatmap Visualizer")

                sal_slice_c1, sal_slice_c2 = st.columns([1, 2])
                with sal_slice_c1:
                    sal_plane = st.selectbox("View Plane", ["Axial", "Coronal", "Sagittal"], key="sal_plane")
                    sal_bg = st.selectbox("Background Scan", ["T1-CE", "T1", "T2", "FLAIR"], key="sal_bg")
                    bg_k = "t1ce" if sal_bg == "T1-CE" else ("t1" if sal_bg == "T1" else ("t2" if sal_bg == "T2" else "flair"))

                    (z0, z1), (y0, y1), (x0, x1) = saliency_res.crop_bbox
                    st.info(f"📍 Analyzed Crop ROI Bounding Box:\nZ: [{z0}, {z1}], Y: [{y0}, {y1}], X: [{x0}, {x1}]")

                    max_s = mod_data[bg_k].shape[2] - 1 if sal_plane == "Axial" else (mod_data[bg_k].shape[1] - 1 if sal_plane == "Coronal" else mod_data[bg_k].shape[0] - 1)
                    center_s = (z0 + z1) // 2 if sal_plane == "Axial" else ((y0 + y1) // 2 if sal_plane == "Coronal" else (x0 + x1) // 2)
                    s_idx = st.slider("Slice Index", 0, max_s, center_s, key="sal_s_idx")

                with sal_slice_c2:
                    mri_s = get_2d_slice(mod_data[bg_k], sal_plane, s_idx)
                    sal_s = get_2d_slice(saliency_res.saliency_map_3d, sal_plane, s_idx)

                    sal_overlay = create_slice_overlay(
                        mri_slice=mri_s,
                        saliency_slice=sal_s,
                        show_saliency=True,
                        alpha=0.6,
                    )

                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(sal_overlay, origin="lower")
                    ax.axis("off")
                    ax.set_title(f"Gradient Saliency Map • {sal_plane} Slice {s_idx}", color=COLOR_PRIMARY_NAVY)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                st.caption(f"ℹ️ **Methodology Note:** {saliency_res.limitations}")

        # TAB 2: MODALITY ABLATION
        with xai_tabs[1]:
            st.markdown("### Modality Ablation Sensitivity Analysis")
            st.write(
                "Systematically suppresses each MRI channel to zero to empirically measure the "
                "degradation in detected tumor volume and model confidence."
            )

            abl_c1, abl_c2 = st.columns([1, 2])
            with abl_c1:
                abl_sub = st.selectbox("Subregion for Ablation Test", ["Whole Tumor (WT)", "Enhancing Tumor (ET)", "Tumor Core (TC)"])
                abl_code = "WT" if "WT" in abl_sub else ("ET" if "ET" in abl_sub else "TC")

                if st.button("🧪 Run Modality Ablation Test", type="primary", use_container_width=True):
                    with st.spinner(f"Running systematic ablation on {abl_sub}..."):
                        input_tensor, _, _ = prepare_input_tensor(mod_data, device=device)
                        abl_rep = run_modality_ablation(
                            model=model,
                            input_tensor=input_tensor,
                            target_subregion=abl_code,
                            reference_mask=seg_res.binary_masks[1],
                            device=device,
                        )
                        st.session_state["ablation_report"] = abl_rep
                        st.success("✅ Modality ablation completed!")

            abl_rep = st.session_state.get("ablation_report")

            with abl_c2:
                if abl_rep is not None:
                    # Ablation bar chart
                    names = [e.modality for e in abl_rep.results]
                    drops = [e.volume_drop_percent for e in abl_rep.results]

                    fig_abl = go.Figure(
                        data=[
                            go.Bar(
                                x=names,
                                y=drops,
                                marker_color="#ef4444",
                                text=[f"-{d:.1f}%" for d in drops],
                                textposition="auto",
                            )
                        ]
                    )
                    fig_abl.update_layout(
                        title=f"Pathology Volume Drop (%) When Modality Is Omitted ({abl_rep.target_subregion})",
                        yaxis_title="Volume Loss (%)",
                        xaxis_title="Omitted Modality",
                        height=280,
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    st.plotly_chart(fig_abl, use_container_width=True)

            if abl_rep is not None:
                st.markdown("---")
                st.markdown("#### Quantitative Ablation Results")
                
                table_rows = []
                for entry in abl_rep.results:
                    table_rows.append(
                        {
                            "Importance Rank": f"#{entry.importance_rank}",
                            "Omitted Modality": entry.modality,
                            "Baseline Voxels": entry.baseline_volume_voxels,
                            "Ablated Voxels": entry.ablated_volume_voxels,
                            "Volume Loss (%)": f"{entry.volume_drop_percent:.1f}%",
                            "Confidence Drop": f"{entry.confidence_drop:.4f}",
                        }
                    )
                st.dataframe(table_rows, use_container_width=True)
                st.info(f"💡 **Clinical Finding:** {abl_rep.summary_text}")

        # TAB 3: SCIENTIFIC EXPLANATION
        with xai_tabs[2]:
            st.markdown("### Clinical Interpretation & Scientific Principles")
            st.markdown(
                """
                #### Why Multimodal Fusion Matters
                In neuro-oncology, different tumor compartments possess distinct biophysical properties that manifest in MRI:
                
                1. **Enhancing Tumor (ET) & T1c:**
                   The blood-brain barrier (BBB) is compromised in high-grade glioblastoma, permitting gadolinium chelates to extravasate. 
                   Consequently, **T1c** provides the dominant gradient attribution and volume sensitivity for Enhancing Tumor.
                   
                2. **Peritumoral Edema & T2/FLAIR:**
                   Vasogenic edema consists of extracellular fluid traversing white matter tracts. On native T1, edema is faint hypointensity. 
                   On **FLAIR**, ventricular CSF is suppressed while interstitial edema appears hyperintense, making FLAIR the most critical 
                   sequence for **Whole Tumor (WT)** boundary determination.
                   
                3. **Tumor Core (TC) & T1/T2:**
                   The necrotic core features liquefactive necrosis (high water content, low cellularity). Combining native **T1** hypointensity 
                   with **T2** hyperintensity and non-enhancing borders identifies the complete surgical resection target.

                #### Mathematical Formulation of XAI Attributions
                - **First-Order Input Gradient:**
                  $$\text{Saliency}(x) = \left| \frac{\partial S_c(X)}{\partial X(x)} \right|$$
                - **Modality Attribution Weight:**
                  $$A_m = \frac{1}{|V_{\text{ROI}}|} \sum_{v \in V_{\text{ROI}}} \left| \frac{\partial S_c}{\partial X_m(v)} \right|$$
                - **Documented Limitations:**
                  Local gradient attribution assumes first-order Taylor expansion linearity. Highly non-linear deep convolutional interactions 
                  and long-range skip connections may not be fully captured by first-order gradients alone.
                """
            )


# =============================================================================
# 5. EVALUATION & METRICS PAGE
# =============================================================================

elif page == "📊 Evaluation & Metrics":
    st.markdown("## 📊 Ground-Truth Evaluation & Quantitative Metrics")
    st.write(
        "Upload verified expert manual ground-truth segmentation masks to evaluate the model's performance "
        "using rigorous academic standards: **Dice Similarity Coefficient (DSC)**, **IoU**, **Sensitivity**, and **HD95**."
    )

    seg_res: Optional[SegmentationResult] = st.session_state.get("segmentation_result")
    gt_map = st.session_state.get("ground_truth_map")

    if seg_res is None:
        st.warning("Please run segmentation in **'MRI Upload & Validation'** before evaluating against ground truth.")
    else:
        st.markdown("#### Ground-Truth Label Source")
        gt_col1, gt_col2 = st.columns([1, 1])

        with gt_col1:
            st.write("Option 1: Upload a ground-truth NIfTI label file (`.nii` or `.nii.gz`):")
            uploaded_gt = st.file_uploader("Upload ground truth mask", type=["nii", "nii.gz"], key="up_gt")
            if uploaded_gt is not None:
                gt_img = load_nifti_from_source(uploaded_gt)
                st.session_state["ground_truth_map"] = gt_img.get_fdata().astype(np.uint8)
                st.session_state["ground_truth_evaluated"] = False
                st.success("✅ Ground truth uploaded!")

        with gt_col2:
            st.write("Option 2: Use pre-loaded ground truth (if using Sample Patient):")
            if gt_map is not None:
                st.info(f"Ground truth already loaded for active patient: shape {gt_map.shape}")
            else:
                st.write("No ground truth currently attached.")

        # Trigger Evaluation
        if st.session_state.get("ground_truth_map") is not None:
            if st.button("📈 Compute Quantitative Metrics", type="primary"):
                with st.spinner("Calculating Dice, IoU, Sensitivity, Specificity, and HD95 surface distances..."):
                    gt_subregions = decode_ground_truth_subregions(st.session_state["ground_truth_map"])
                    pred_subregions = {
                        "WT": seg_res.binary_masks[1],
                        "TC": seg_res.binary_masks[0],
                        "ET": seg_res.binary_masks[2],
                    }

                    spacing = st.session_state.get("validation_result").common_spacing or (1.0, 1.0, 1.0)
                    eval_report = evaluate_prediction(pred_subregions, gt_subregions, voxel_spacing=spacing)
                    st.session_state["evaluation_report"] = eval_report
                    st.session_state["ground_truth_evaluated"] = True
                    st.success("✅ Metrics computed successfully!")

        # Display Evaluation Cards and Tables
        eval_report = st.session_state.get("evaluation_report")
        if eval_report is not None and st.session_state.get("ground_truth_evaluated"):
            st.markdown("---")
            st.markdown("### 🏆 Quantitative Academic Evaluation Results")

            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            with m_c1:
                st.markdown(
                    f"""
                    <div class="metric-card" style="border-left-color: #0d9488;">
                        <div class="metric-title">Mean Dice Score</div>
                        <div class="metric-value">{eval_report.mean_dice:.4f}</div>
                        <div class="metric-sub">Across WT, TC, and ET</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m_c2:
                st.markdown(
                    f"""
                    <div class="metric-card" style="border-left-color: #0f2b48;">
                        <div class="metric-title">Mean IoU (Jaccard)</div>
                        <div class="metric-value">{eval_report.mean_iou:.4f}</div>
                        <div class="metric-sub">Overlap intersection</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m_c3:
                st.markdown(
                    f"""
                    <div class="metric-card" style="border-left-color: #f59e0b;">
                        <div class="metric-title">Mean Sensitivity</div>
                        <div class="metric-value">{eval_report.mean_sensitivity:.4f}</div>
                        <div class="metric-sub">True Positive Detection</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m_c4:
                hd_display = f"{eval_report.mean_hd95:.2f} mm" if eval_report.mean_hd95 is not None else "N/A"
                st.markdown(
                    f"""
                    <div class="metric-card" style="border-left-color: #ef4444;">
                        <div class="metric-title">Mean HD95 Distance</div>
                        <div class="metric-value">{hd_display}</div>
                        <div class="metric-sub">95th Percentile Boundary</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Detailed metrics table
            st.markdown("#### Subregion Breakdown Table")
            metric_data = []
            for sub_name in ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"]:
                key = "WT" if "WT" in sub_name else ("TC" if "TC" in sub_name else "ET")
                m = eval_report.metrics_per_subregion[key]
                hd_str = f"{m.hd95:.2f} mm" if (m.hd95 is not None and np.isfinite(m.hd95)) else "Undefined"
                metric_data.append(
                    {
                        "Subregion": sub_name,
                        "Dice Score (DSC)": f"{m.dice:.4f}",
                        "IoU (Jaccard)": f"{m.iou:.4f}",
                        "Sensitivity (Recall)": f"{m.sensitivity:.4f}",
                        "Specificity": f"{m.specificity:.4f}",
                        "HD95 (mm)": hd_str,
                        "Pred Voxels": f"{m.pred_voxels:,}",
                        "Truth Voxels": f"{m.truth_voxels:,}",
                    }
                )
            st.dataframe(metric_data, use_container_width=True)

            st.caption(f"ℹ️ {eval_report.empty_mask_policy}")

            # Visual Comparison of Prediction vs Ground Truth
            st.markdown("---")
            st.markdown("#### Visual Comparison: Model Prediction vs Ground Truth")

            comp_c1, comp_c2 = st.columns(2)
            gt_m = st.session_state["ground_truth_map"]
            mid_z = gt_m.shape[2] // 2
            comp_slice = st.slider("Select Slice to Compare", 0, gt_m.shape[2] - 1, mid_z)

            # Extract slices
            pred_wt_2d = seg_res.binary_masks[1][:, :, comp_slice].T
            truth_sub = decode_ground_truth_subregions(gt_m)
            gt_wt_2d = truth_sub["WT"][:, :, comp_slice].T
            mri_bg = st.session_state["modalities_data"]["flair"][:, :, comp_slice].T

            with comp_c1:
                ov_pred = create_slice_overlay(mri_bg, wt_slice=pred_wt_2d, show_wt=True, alpha=0.5)
                fig_p, ax_p = plt.subplots(figsize=(5, 5))
                ax_p.imshow(ov_pred, origin="lower")
                ax_p.axis("off")
                ax_p.set_title(f"Model Prediction (WT) • Slice {comp_slice}", color=COLOR_PRIMARY_NAVY)
                st.pyplot(fig_p, use_container_width=True)
                plt.close(fig_p)

            with comp_c2:
                ov_gt = create_slice_overlay(mri_bg, wt_slice=gt_wt_2d, show_wt=True, alpha=0.5)
                fig_g, ax_g = plt.subplots(figsize=(5, 5))
                ax_g.imshow(ov_gt, origin="lower")
                ax_g.axis("off")
                ax_g.set_title(f"Ground Truth (WT) • Slice {comp_slice}", color=COLOR_PRIMARY_NAVY)
                st.pyplot(fig_g, use_container_width=True)
                plt.close(fig_g)


# =============================================================================
# 6. METHODOLOGY PAGE
# =============================================================================

elif page == "📖 Methodology":
    st.markdown("## 📖 Deep Learning & XAI Methodology")
    st.write("A comprehensive walkthrough of the computational, anatomical, and mathematical foundations.")

    st.markdown(
        """
        ### 1. Multimodal MRI Input
        Brain tumor segmentation requires 4 core MRI sequences:
        - **T1c (T1-CE):** T1-weighted scan with gadolinium contrast agent. Enhancing malignant margins stand out clearly.
        - **T1:** Native unenhanced T1-weighted scan. Highlights normal gray and white matter anatomy.
        - **T2:** T2-weighted scan. Hyperintense for liquid and fluid compartments.
        - **FLAIR:** Fluid-Attenuated Inversion Recovery. Suppresses normal ventricular CSF to selectively highlight peritumoral vasogenic edema.
        
        ### 2. Validation & Geometry Checks
        Before inference, all inputs undergo rigorous integrity verification:
        - Format validation (.nii or .nii.gz format; rejects 2D PNG/JPG screenshots).
        - Dimensionality: Verifies genuine 3D volumetric arrays $(X, Y, Z)$.
        - Spatial grid alignment: Confirms matching shapes and checks affine orientation matrix alignment within clinical tolerance.
        
        ### 3. Preprocessing Pipeline
        - **Channel Stacking:** Modalities are ordered strictly per MONAI BraTS specification:
          `[Channel 0: T1c, Channel 1: T1, Channel 2: T2, Channel 3: FLAIR]`.
        - **Intensity Normalization:** Non-zero channel-wise intensity standardization:
          $$I_{\text{norm}}(v) = \frac{I(v) - \mu_{\text{nonzero}}}{\sigma_{\text{nonzero}}}$$
        - **Symmetric Spatial Padding:** Padded symmetrically so that spatial dimensions are divisible by 16 for UNet-style downsampling.
        
        ### 4. Backbone Network: MONAI SegResNet
        The model uses **SegResNet** (Myronenko 2018), the winner of the BraTS challenge:
        - An asymmetric encoder-decoder with residual building blocks and group normalization.
        - Downsampling blocks: `[1, 2, 2, 4]` with 16 initial filters.
        - Upsampling blocks: `[1, 1, 1]`.
        - Inference uses **3D Sliding Window Inferer** with Gaussian blending weights to eliminate boundary seam artifacts.
        
        ### 5. Multi-Class Label Decoding
        The network outputs 3 independent logits $(C_{\text{TC}}, C_{\text{WT}}, C_{\text{ET}})$.
        Sigmoid activation is applied: $P_c = \sigma(L_c) = \frac{1}{1 + e^{-L_c}}$, thresholded at $0.5$.
        To construct the mutually exclusive BraTS label map:
        $$\text{Label}(v) = \begin{cases} 
        4 & \text{if } P_{\text{ET}}(v) \ge 0.5 \quad \text{(Enhancing Tumor)} \\
        1 & \text{else if } P_{\text{TC}}(v) \ge 0.5 \quad \text{(Necrotic / Non-enhancing Core)} \\
        2 & \text{else if } P_{\text{WT}}(v) \ge 0.5 \quad \text{(Peritumoral Edema)} \\
        0 & \text{otherwise} \quad \text{(Background)}
        \end{cases}$$
        
        ### 6. Explainable AI (XAI)
        - **Gradient Saliency:** Backpropagates output target score to input voxels to calculate $\frac{\partial S}{\partial X}$.
        - **Modality Attribution:** Averages gradient magnitude per channel to compute the relative percentage contribution of each sequence.
        - **Modality Ablation:** Empirically zeros out each channel and recalculates pathology volume loss and confidence drop.
        
        ### 7. Evaluation Metrics
        Evaluated on common spatial grids:
        - **Dice Similarity Coefficient (DSC):** $2|P \cap T| / (|P| + |T|)$
        - **Intersection over Union (IoU):** $|P \cap T| / (|P \cup T|)$
        - **Sensitivity / Recall:** $TP / (TP + FN)$
        - **Hausdorff Distance 95% (HD95):** 95th percentile Euclidean boundary distance in millimeters.
        """
    )


# =============================================================================
# 7. ABOUT & ACADEMIC HONESTY PAGE
# =============================================================================

elif page == "ℹ️ About & Honesty":
    st.markdown("## ℹ️ About & Academic Integrity Statement")

    st.markdown(
        """
        ### Technology Stack
        - **Language & Framework:** Python 3.13, Streamlit
        - **Deep Learning:** PyTorch, MONAI (Medical Open Network for AI)
        - **Neuroimaging:** NiBabel, NumPy, SciPy
        - **Visualization:** Matplotlib, Plotly

        ### Pretrained Model Source
        - **Bundle:** `brats_mri_segmentation` (Version 0.5.4)
        - **Authors:** MONAI Team & Consortium
        - **Backbone Architecture:** SegResNet (Myronenko 2018)
        - **Weights URL:** Hosted on HuggingFace (`MONAI/brats_mri_segmentation`) & NVIDIA Clara Model Zoo.
        - **Validation Dice Performance:** Whole Tumor: 0.9026, Tumor Core: 0.8559, Enhancing Tumor: 0.7905.
        
        ---
        
        ### 🎓 Academic Honesty & MGAF Research Disclosure
        > [!IMPORTANT]
        > **Academic Transparency:**
        > The initial project proposal explored **Modality-Gated Attention Fusion (MGAF)** and a baseline 3D U-Net. 
        > This working implementation utilizes the official pretrained **MONAI SegResNet** backbone to deliver a robust, 
        > validated, and clinically functional application for project review.
        > 
        > **MGAF is presented as the proposed future research direction** to introduce channel-level attention gates into 
        > multimodal feature fusion, rather than falsely claiming that SegResNet is MGAF or claiming to have trained 
        > a novel model from scratch.

        ---

        ### References
        1. **Myronenko, A.** (2018). *3D MRI brain tumor segmentation using autoencoder regularization*. International MICCAI Brainlesion Workshop, Springer, Cham, 311-320.
        2. **Bakas, S., et al.** (2018). *Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge*. arXiv:1811.02629.
        3. **Cardoso, M. J., et al.** (2022). *MONAI: An open-source framework for deep learning in healthcare*. arXiv:2211.02701.

        ---

        ### ⚠️ Clinical Disclaimer
        <div class="disclaimer-box">
            <strong>RESEARCH & EDUCATIONAL USE ONLY:</strong> This software and deep-learning segmentation pipeline is designed 
            strictly for academic demonstration, research experimentation, and engineering project review. It has NOT been 
            cleared or approved by the FDA or CE for clinical diagnosis, surgical navigation, radiation therapy, or patient management.
        </div>
        """,
        unsafe_allow_html=True,
    )
