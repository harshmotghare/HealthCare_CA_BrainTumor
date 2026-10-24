# Multimodal Brain Tumor Segmentation Using MRI Data and Explainable AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: MONAI](https://img.shields.io/badge/framework-MONAI%201.4+-green.svg)](https://monai.io/)
[![Deep Learning: PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Dashboard: Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Tests: Pytest](https://img.shields.io/badge/tests-18%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

An end-to-end, clinically motivated deep-learning web application for 3D multi-sequence brain tumor segmentation and Explainable AI (XAI) attribution. Built with **MONAI**, **PyTorch**, **NiBabel**, and **Streamlit**.

---

## 📌 Table of Contents
1. [Executive Summary](#executive-summary)
2. [Clinical Motivation & Modality Semantics](#clinical-motivation--modality-semantics)
3. [Deep Learning Architecture: SegResNet](#deep-learning-architecture-segresnet)
4. [Explainable AI (XAI) Framework](#explainable-ai-xai-framework)
5. [Evaluation Metrics & Ground-Truth Verification](#evaluation-metrics--ground-truth-verification)
6. [Academic Honesty & MGAF Research Disclosure](#academic-honesty--mgaf-research-disclosure)
7. [Project Directory Structure](#project-directory-structure)
8. [Windows Setup & Execution Guide](#windows-setup--execution-guide)
9. [Running Tests](#running-tests)
10. [Launching the Web Application](#launching-the-web-application)
11. [Clinical Disclaimer](#clinical-disclaimer)


---

## 1. Executive Summary

Gliomas are aggressive primary brain neoplasms characterized by heterogeneous infiltration. Accurate volumetric delineation of tumor subcompartments is vital for neurosurgical resection planning, radiotherapy target definition, and longitudinal oncology assessment.

This application provides a production-grade, interactive dashboard that:
- Ingests and validates four 3D MRI sequences (**T1**, **T1-CE / T1c**, **T2**, and **T2-FLAIR**).
- Applies nonzero channel-wise standardization and spatial tensor padding.
- Runs sliding-window volumetric inference using the official pretrained **MONAI `brats_mri_segmentation` SegResNet** backbone.
- Extracts mutually exclusive tumor subregions: **Whole Tumor (WT)**, **Tumor Core (TC)**, and **Enhancing Tumor (ET)**.
- Computes genuine **backpropagation gradient saliency maps** and **modality ablation sensitivity analysis** for Explainable AI.
- Evaluates predictions against expert ground truth using **Dice**, **IoU**, **Sensitivity**, and **95th percentile Hausdorff Distance (HD95)**.
- Exports segmentations as spatially preserved 3D NIfTI volumes with reference affine matrices.

---

## 2. Clinical Motivation & Modality Semantics

### Input Sequences & Channel Ordering
The application strictly respects the official MONAI BraTS input channel specification:

| Channel | Modality | Clinical Radiological Significance |
| :---: | :--- | :--- |
| **0** | **T1c (T1-CE)** | Contrast-enhanced T1 with gadolinium chelates; highlights vascular extravasation and viable malignant margin (**ET**). |
| **1** | **T1** | Native unenhanced T1; highlights anatomical boundaries between gray matter, white matter, and necrotic fluid. |
| **2** | **T2** | T2-weighted; hyperintense for fluids and pathologically altered tissue. |
| **3** | **FLAIR** | CSF-suppressed T2; selectively highlights vasogenic **peritumoral edema** across axonal tracts. |

### Output Subregion Definitions
The network generates three independent sigmoid probability maps thresholded at $0.5$:
- **Whole Tumor (WT):** Peritumoral edema + necrotic core + active enhancing tumor.
- **Tumor Core (TC):** Resectable solid mass (necrotic/non-enhancing core + enhancing margin).
- **Enhancing Tumor (ET):** Active contrast-accumulating rim.

#### Multi-Class Discrete Mapping (BraTS Standard)
$$\text{Label}(v) = \begin{cases} 
4 & \text{if } P_{\text{ET}}(v) \ge 0.5 \quad \text{(Enhancing Tumor)} \\
1 & \text{else if } P_{\text{TC}}(v) \ge 0.5 \quad \text{(Necrotic / Non-enhancing Core)} \\
2 & \text{else if } P_{\text{WT}}(v) \ge 0.5 \quad \text{(Peritumoral Edema)} \\
0 & \text{otherwise} \quad \text{(Background)}
\end{cases}$$

---

## 3. Deep Learning Architecture: SegResNet

The model utilizes **SegResNet** (Myronenko, MICCAI BraTS 2018 winning architecture):
- **Encoder:** Asymmetric encoder-decoder with residual blocks and Group Normalization.
- **Downsampling Layers:** `[1, 2, 2, 4]` blocks with 16 initial filters.
- **Upsampling Layers:** `[1, 1, 1]` residual upsampling blocks.
- **Parameters:** 1,180,483 parameters.
- **Inference Mode:** 3D `SlidingWindowInferer` with Gaussian window blending to eliminate edge tiling artifacts.
- **Pretrained Benchmark Validation Dice:**
  - Tumor Core (TC): **0.8559**
  - Whole Tumor (WT): **0.9026**
  - Enhancing Tumor (ET): **0.7905**
  - Overall Mean: **0.8518**

---

## 4. Explainable AI (XAI) Framework

Unlike decorative or arbitrary heatmaps, this project computes **genuine mathematical attribution**:

### 1. Backpropagation Gradient Saliency
Attaches gradient tracking to the input tensor and computes:
$$\text{Saliency}(x) = \left| \frac{\partial S_c(X)}{\partial X(x)} \right|$$
where $S_c$ is the scalar logit score for the target subregion (WT, TC, or ET).
- **Per-Modality Attribution Weight:** Quantifies the average gradient magnitude per channel:
  $$A_m = \frac{1}{|V_{\text{ROI}}|} \sum_{v \in V_{\text{ROI}}} \left| \frac{\partial S_c}{\partial X_m(v)} \right|$$
- Identifies which imaging sequence was most influential (e.g., T1c dominates for ET; FLAIR dominates for WT).

### 2. Systematic Modality Ablation
Suppresses each modality to zero individually and re-runs inference:
- Quantifies volume loss percentage: $\Delta V = \frac{V_{\text{base}} - V_{\text{abl}}}{V_{\text{base}}} \times 100\%$
- Quantifies confidence drop: $\Delta P = \bar{P}_{\text{base}} - \bar{P}_{\text{abl}}$
- Ranks sequences by empirical sensitivity.

### 3. Documented Limitations
- Local gradient attribution is a first-order Taylor approximation and does not capture higher-order non-linear feature interactions.
- Patch-based ROI analysis may omit contextual cues present across the broader field of view.

---

## 5. Evaluation Metrics & Ground-Truth Verification

When expert ground-truth annotations are provided, the system calculates:
- **Dice Similarity Coefficient (DSC):** $2|P \cap T| / (|P| + |T|)$
- **Intersection over Union (IoU / Jaccard):** $|P \cap T| / (|P \cup T|)$
- **Sensitivity / Recall:** $TP / (TP + FN)$
- **Specificity:** $TN / (TN + FP)$
- **95th Percentile Hausdorff Distance (HD95):** Surface boundary distance in millimeters calculated using 3D Euclidean Distance Transforms (`scipy.ndimage.distance_transform_edt`).

### Empty Mask Policy
- If both prediction and ground truth are empty for a subregion: $\text{DSC}=1.0$, $\text{IoU}=1.0$, $\text{Sensitivity}=1.0$, and $\text{HD95}=0.0\text{ mm}$.
- If one is empty and the other contains voxels: overlap metrics are $0.0$ and HD95 is reported as undefined / infinite.

---

## 6. Academic Honesty & MGAF Research Disclosure

> [!IMPORTANT]
> **Academic Integrity Statement:**
> Earlier project proposals outlined **Modality-Gated Attention Fusion (MGAF)** with a baseline 3D U-Net. 
> To deliver a working, robust, and verifiable software product for engineering review without fabricating experimental 
> training results, this implementation utilizes the official pretrained **MONAI SegResNet** backbone.
> 
> **MGAF is presented as the proposed future research direction** to introduce channel-level attention gates into 
> multimodal feature fusion, rather than misrepresenting SegResNet as MGAF or claiming to have trained a custom novel network.

---

## 7. Project Directory Structure

```
HealthCareProjF/
├── app.py                      # Main multi-page Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation
├── .gitignore                  # Git exclusions for checkpoints and logs
│
├── models/
│   ├── pretrained_model.py     # SegResNet loader and MONAI bundle manager
│   └── brats_mri_segmentation/ # Official MONAI bundle configs and metadata
│
├── data/
│   ├── validation.py           # 3D NIfTI format, spacing, and affine checks
│   └── preprocessing.py        # Channel ordering, normalization, spatial padding
│
├── inference/
│   ├── predictor.py            # Sliding-window inference and subregion decoding
│   └── export.py               # Spatial affine preservation and NIfTI export
│
├── explainability/
│   ├── saliency.py             # Backpropagation gradient saliency engine
│   └── modality_ablation.py    # Sequence omission sensitivity analysis
│
├── evaluation/
│   └── metrics.py              # Dice, IoU, Sensitivity, Specificity, HD95
│
├── ui/
│   └── styles.py               # Clinical navy/teal styling and custom CSS
│
├── tests/
│   ├── test_validation.py      # Input validation test suite
│   ├── test_preprocessing.py   # Tensor stacking and padding tests
│   ├── test_model.py           # SegResNet instantiation & checkpoint tests
│   ├── test_metrics.py         # Metric calculation and ground truth tests
│   └── test_export.py          # NIfTI affine & byte roundtrip tests
│
├── artifacts/
│   └── checkpoints/            # Downloaded model weights (model.pt)
│
└── data_local/
    ├── README.md               # Instructions for obtaining BraTS dataset
    └── sample_patient/         # Pre-packaged 3D phantom & ground truth
```

---

## 8. Windows Setup & Execution Guide

### Prerequisites
- Windows 10/11
- Python 3.10 to 3.13
- PowerShell or Windows Terminal

### Step-by-Step Setup

```powershell
# 1. Clone or navigate to the workspace
cd C:\Users\LENOVO\Desktop\HealthCareProjF

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 9. Running Tests

Run the comprehensive unit test suite to verify model loading, preprocessing, validation, metrics, and spatial export:

```powershell
python -m pytest tests -v
```

Expected output:
```
tests/test_export.py::test_spatial_consistency_verification PASSED
tests/test_export.py::test_nifti_image_creation_and_affine PASSED
tests/test_export.py::test_export_mask_to_bytes_roundtrip PASSED
tests/test_metrics.py::test_dice_identical_masks PASSED
tests/test_metrics.py::test_dice_disjoint_masks PASSED
tests/test_metrics.py::test_empty_mask_policy PASSED
tests/test_metrics.py::test_ground_truth_decoding_brats_labels PASSED
tests/test_metrics.py::test_hd95_known_geometry PASSED
tests/test_model.py::test_model_architecture_definition PASSED
tests/test_model.py::test_checkpoint_discovery PASSED
tests/test_model.py::test_load_pretrained_model PASSED
tests/test_preprocessing.py::test_channel_order_preservation PASSED
tests/test_preprocessing.py::test_pad_and_unpad_symmetry PASSED
tests/test_preprocessing.py::test_prepare_input_tensor_pipeline PASSED
tests/test_validation.py::test_validation_all_modalities_valid PASSED
tests/test_validation.py::test_validation_missing_modality PASSED
tests/test_validation.py::test_validation_shape_mismatch PASSED
tests/test_validation.py::test_validation_screenshot_rejection PASSED
============================= 18 passed in 14.24s =============================
```

---

## 10. Launching the Web Application

To launch the Streamlit clinical dashboard:

```powershell
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

### Instant Demo Instructions:
1. Navigate to **"📤 MRI Upload & Validation"** in the sidebar.
2. Click **"🚀 Load Pre-packaged Sample Patient"**.
3. Click **"🚀 Run Segmentation"**.
4. Explore **"🧠 Segmentation Results"**, **"🔬 Explainable AI"**, and **"📊 Evaluation & Metrics"**!

---

## 11. Clinical Disclaimer

> [!CAUTION]
> **RESEARCH & EDUCATIONAL USE ONLY:**
> This software is intended strictly for academic research, engineering education, and algorithmic evaluation. 
> It has **not** been approved by the US FDA, European CE, or any regulatory medical authority for clinical diagnostic, 
> radiological, surgical, or treatment planning purposes.
