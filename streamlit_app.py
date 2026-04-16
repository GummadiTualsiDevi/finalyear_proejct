"""
streamlit_app.py — Streamlit frontend for WSI Cancer Detection.

Provides a premium web interface to:
  • Upload whole slide images (WSI) or standard images
  • Preview slide thumbnails with metadata
  • Run ViT-based cancer detection analysis
  • Display tumor probability heatmaps and overlays
  • Show prediction results, suspicious region coordinates
  • Download heatmap overlay images

Usage:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import time
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.config import (
    DISCLAIMER, CLASS_NAMES, get_device, DEFAULT_PATCH_SIZE,
    DEFAULT_STRIDE, DEFAULT_TISSUE_THRESHOLD, DEFAULT_BATCH_SIZE,
    is_wsi_file, is_image_file, IMAGE_SIZE, VIT_MODEL_NAME,
)
from utils.slide_utils import load_slide, SlideWrapper
from models.load_pretrained_model import load_model_auto
from data.patch_extractor import extract_patches
from inference.patch_inference import run_batch_inference
from inference.slide_prediction import (
    build_probability_grid, classify_slide, get_suspicious_regions,
    get_prediction_summary,
)
from visualization.heatmap_generator import (
    generate_slide_heatmap, highlight_tumor_regions,
    encode_image_to_bytes, generate_matplotlib_heatmap,
)


# ═══════════════════════════════════════════════════
#  PAGE CONFIG & CUSTOM CSS
# ═══════════════════════════════════════════════════

st.set_page_config(
    page_title="WSI Cancer Detection — ViT AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 16px;
        color: white;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        font-size: 1rem;
        opacity: 0.8;
        margin-top: 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 14px;
        color: white;
        text-align: center;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.85rem;
        opacity: 0.85;
        font-weight: 400;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card p {
        margin: 0.4rem 0 0 0;
        font-size: 1.5rem;
        font-weight: 700;
    }

    .tumor-detected {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%) !important;
        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.4) !important;
    }
    .no-tumor {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%) !important;
        box-shadow: 0 4px 15px rgba(39, 174, 96, 0.4) !important;
    }

    .disclaimer-box {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border-left: 5px solid #f39c12;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        color: #7d6608;
        font-size: 0.9rem;
    }

    .info-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }

    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #3498db;
        display: inline-block;
    }

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1a2e, #16213e);
    }

    .region-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    .region-table th {
        background: #3498db;
        color: white;
        padding: 0.5rem;
        text-align: left;
    }
    .region-table td {
        padding: 0.4rem 0.5rem;
        border-bottom: 1px solid #eee;
    }
    .region-table tr:hover {
        background: #f0f7ff;
    }

    .risk-very-high { color: #c0392b; font-weight: 700; }
    .risk-high { color: #e74c3c; font-weight: 600; }
    .risk-moderate { color: #f39c12; font-weight: 500; }
    .risk-low { color: #27ae60; }

    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.85rem;
        padding: 1.5rem 0;
        margin-top: 2rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>🔬 WSI Cancer Detection</h1>
    <p>Vision Transformer (ViT) based metastatic cancer detection in lymph node histopathology slides</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f'<div class="disclaimer-box">{DISCLAIMER}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
#  MODEL LOADING (cached)
# ═══════════════════════════════════════════════════

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pth")


@st.cache_resource
def load_cached_model():
    """Load or initialize the ViT model once, cached across Streamlit reruns."""
    device = get_device()
    model, device, source = load_model_auto(MODEL_PATH, device)
    return model, device, source


model, device, model_source = load_cached_model()

# Show model status
if "pretrained" in model_source.lower() and "checkpoint" not in model_source.lower():
    st.info(
        "🔄 **Using pretrained ImageNet weights** (no fine-tuned checkpoint found at "
        f"`{MODEL_PATH}`).  \n"
        "For accurate cancer detection, train the model first:  \n"
        "`python train.py --data_dir dataset --epochs 10 --batch_size 32`"
    )


# ═══════════════════════════════════════════════════
#  SIDEBAR — SETTINGS
# ═══════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Analysis Settings")

    patch_size = st.slider(
        "Patch Size (px)", 96, 448, DEFAULT_PATCH_SIZE, step=32,
        help="Size of each analysis patch extracted from the slide"
    )
    stride = st.slider(
        "Stride (px)", 56, 448, DEFAULT_STRIDE, step=28,
        help="Sliding window step size (smaller = more overlap, more patches)"
    )
    tissue_thresh = st.slider(
        "Tissue Threshold", 0.1, 0.8, DEFAULT_TISSUE_THRESHOLD, step=0.05,
        help="Minimum tissue fraction required to analyse a patch"
    )
    batch_size = st.slider(
        "Batch Size", 4, 64, DEFAULT_BATCH_SIZE, step=4,
        help="Number of patches processed simultaneously (larger = faster on GPU)"
    )

    aggregation = st.selectbox(
        "Aggregation Method",
        ["combined", "max", "mean", "top_k"],
        help="How to combine patch predictions into a slide-level score"
    )

    st.markdown("---")
    st.markdown(f"**🧠 Model:** `{VIT_MODEL_NAME}`")
    st.markdown(f"**📍 Source:** `{model_source.split(':')[0]}`")
    st.markdown(f"**💻 Device:** `{device}`")
    st.markdown(f"**📐 Input:** `{IMAGE_SIZE}×{IMAGE_SIZE}`")

    st.markdown("---")
    st.markdown("### 📚 Supported Formats")
    st.markdown("**WSI:** `.svs`, `.tif`, `.tiff`, `.ndpi`")
    st.markdown("**Image:** `.jpg`, `.jpeg`, `.png`")


# ═══════════════════════════════════════════════════
#  FILE UPLOAD
# ═══════════════════════════════════════════════════

st.markdown('<div class="section-header">📤 Upload Slide Image</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a histopathology slide image",
    type=["svs", "tif", "tiff", "ndpi", "jpg", "jpeg", "png"],
    help="Upload a Whole Slide Image or a standard histopathology image"
)


# ═══════════════════════════════════════════════════
#  ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════

if uploaded_file is not None:
    # Determine file extension and save to temp
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # ── Slide Preview Section ──
        st.markdown('<div class="section-header">🖼️ Slide Preview</div>', unsafe_allow_html=True)

        slide = load_slide(tmp_path)
        metadata = slide.get_metadata_summary()
        thumbnail = slide.get_thumbnail((1024, 1024))
        thumb_np = np.array(thumbnail)

        col_preview, col_meta = st.columns([2, 1])

        with col_preview:
            st.image(thumb_np, caption=f"Thumbnail — {metadata['filename']}", use_container_width=True)

        with col_meta:
            st.markdown("**📋 Slide Metadata**")
            st.markdown(f"**File:** `{metadata['filename']}`")
            st.markdown(f"**Format:** {metadata['format']}")
            st.markdown(f"**Dimensions:** {metadata['dimensions']}")
            st.markdown(f"**Pyramid Levels:** {metadata['levels']}")

            if metadata.get("magnification"):
                st.markdown(f"**Magnification:** {metadata['magnification']}×")
            if metadata.get("vendor"):
                st.markdown(f"**Vendor:** {metadata['vendor']}")

            for lvl_str in metadata.get("level_dimensions", []):
                st.markdown(f"  `{lvl_str}`")

        # Auto-adjust for very small images
        h, w = thumb_np.shape[:2]
        effective_patch = patch_size
        effective_stride = stride
        if h < 500 or w < 500:
            st.warning(f"⚠️ Low resolution image ({w}×{h}). Auto-adjusting patch size.")
            if effective_patch > min(h, w):
                effective_patch = max(32, min(h, w) // 2)
                effective_stride = max(16, effective_patch // 2)
                st.info(f"Auto-adjusted: Patch={effective_patch}px, Stride={effective_stride}px")

        st.markdown("---")

        # ── Analysis Button ──
        analyze_button = st.button(
            "🔍 Run Cancer Detection Analysis",
            type="primary",
            use_container_width=True,
        )

        if analyze_button:
            st.markdown('<div class="section-header">⏳ Analysis Progress</div>', unsafe_allow_html=True)

            progress_bar = st.progress(0, text="Initializing analysis pipeline...")

            # Step 1: Patch Extraction
            progress_bar.progress(5, text="🔍 Detecting tissue and extracting patches...")

            def extraction_progress(current, total):
                pct = int(5 + (current / max(total, 1)) * 35)
                progress_bar.progress(min(pct, 40), text=f"🧩 Extracting patches: {current}/{total}")

            patches, grid_shape, tissue_mask, slide_wrapper = extract_patches(
                slide_path=tmp_path,
                patch_size=effective_patch,
                stride=effective_stride,
                tissue_threshold=tissue_thresh,
                progress_callback=extraction_progress,
            )

            if len(patches) == 0:
                st.warning("⚠️ No tissue patches found in this image. Try adjusting the tissue threshold.")
                st.stop()

            progress_bar.progress(40, text=f"✅ Extracted {len(patches)} tissue patches")

            # Step 2: Run Inference
            progress_bar.progress(42, text="🧠 Running ViT inference on patches...")

            def inference_progress(current, total):
                pct = int(42 + (current / max(total, 1)) * 40)
                progress_bar.progress(min(pct, 82), text=f"🧠 Inference: batch {current}/{total}")

            probabilities, inference_time = run_batch_inference(
                model=model,
                patches=patches,
                device=device,
                batch_size=batch_size,
                progress_callback=inference_progress,
            )

            progress_bar.progress(82, text="📊 Aggregating predictions...")

            # Step 3: Build probability grid and classify
            prob_grid = build_probability_grid(patches, probabilities, grid_shape)
            prediction_summary = get_prediction_summary(patches, probabilities, method=aggregation)

            progress_bar.progress(88, text="🌡️ Generating heatmap visualization...")

            # Step 4: Generate heatmap
            heatmap_results = generate_slide_heatmap(thumb_np, prob_grid)
            annotated = highlight_tumor_regions(
                thumb_np, prob_grid, patches, probabilities,
                stride=effective_stride, patch_size=effective_patch,
            )

            progress_bar.progress(100, text="✅ Analysis complete!")
            time.sleep(0.5)
            progress_bar.empty()

            # ═══════════════════════════════════════
            #  RESULTS DISPLAY
            # ═══════════════════════════════════════

            st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)

            is_tumor = prediction_summary["prediction"] == "Tumor Detected"
            css_class = "tumor-detected" if is_tumor else "no-tumor"
            icon = "🔴" if is_tumor else "🟢"

            # ── Metric Cards ──
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="metric-card {css_class}">
                    <h3>Prediction</h3>
                    <p>{icon} {prediction_summary['prediction']}</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Slide Cancer Probability</h3>
                    <p>{prediction_summary['slide_prob']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Max Patch Probability</h3>
                    <p>{prediction_summary['max_prob']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Suspicious Patches</h3>
                    <p>{prediction_summary['suspicious']} / {prediction_summary['total_patches']}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")

            # ── Heatmap Visualization ──
            st.markdown('<div class="section-header">🌡️ Heatmap Visualization</div>', unsafe_allow_html=True)

            tab_overlay, tab_heatmap, tab_regions, tab_matplotlib = st.tabs([
                "📷 Overlay", "🌡️ Heatmap Only", "🎯 Tumor Regions", "📊 Probability Map"
            ])

            with tab_overlay:
                col_orig, col_over = st.columns(2)
                with col_orig:
                    st.markdown("##### Original Slide")
                    st.image(thumb_np, use_container_width=True)
                with col_over:
                    st.markdown("##### Heatmap Overlay")
                    st.image(heatmap_results["overlay_rgb"], use_container_width=True)

            with tab_heatmap:
                st.markdown("##### Cancer Probability Heatmap")
                st.image(heatmap_results["heatmap_rgb"], use_container_width=True)
                st.caption("Blue = Low probability | Red = High probability")

            with tab_regions:
                st.markdown("##### Highlighted Tumor Regions")
                st.image(annotated, use_container_width=True)
                st.caption("Red boxes indicate patches with >50% tumor probability")

            with tab_matplotlib:
                st.markdown("##### Probability Distribution Map")
                fig = generate_matplotlib_heatmap(prob_grid)
                st.pyplot(fig)

            # ── Detailed Analysis ──
            st.markdown('<div class="section-header">📋 Analysis Details</div>', unsafe_allow_html=True)

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.markdown(f"""
                | Metric | Value |
                |---|---|
                | Total tissue patches | {prediction_summary['total_patches']} |
                | Suspicious patches (>50%) | {prediction_summary['suspicious']} |
                | High-risk patches (>70%) | {prediction_summary.get('top_suspicious_count', 0)} |
                | Slide cancer probability | {prediction_summary['slide_prob']:.4f} |
                | Max patch probability | {prediction_summary['max_prob']:.4f} |
                | Avg patch probability | {prediction_summary['avg_prob']:.4f} |
                | Confidence level | {prediction_summary['confidence']} |
                | Inference time | {inference_time:.2f}s |
                | Patch size | {effective_patch}×{effective_patch}px |
                | Stride | {effective_stride}px |
                | Aggregation | {aggregation} |
                """)

            with detail_col2:
                # ── Suspicious Region Coordinates ──
                suspicious_regions = prediction_summary.get("suspicious_regions", [])

                if suspicious_regions:
                    st.markdown("**🎯 Suspicious Region Coordinates**")

                    table_html = '<table class="region-table"><tr>'
                    table_html += '<th>#</th><th>Position (X, Y)</th><th>Grid (R, C)</th>'
                    table_html += '<th>Probability</th><th>Risk</th></tr>'

                    for i, region in enumerate(suspicious_regions[:20]):
                        risk = region["risk_level"]
                        risk_class = f"risk-{risk.lower().replace(' ', '-')}"
                        table_html += f'<tr>'
                        table_html += f'<td>{i + 1}</td>'
                        table_html += f'<td>({region["x"]}, {region["y"]})</td>'
                        table_html += f'<td>({region["row"]}, {region["col"]})</td>'
                        table_html += f'<td>{region["tumor_probability"]:.4f}</td>'
                        table_html += f'<td class="{risk_class}">{risk}</td>'
                        table_html += f'</tr>'

                    table_html += '</table>'

                    if len(suspicious_regions) > 20:
                        table_html += f'<p style="color:#888;font-size:0.8rem;">Showing top 20 of {len(suspicious_regions)} suspicious regions</p>'

                    st.markdown(table_html, unsafe_allow_html=True)
                else:
                    st.success("✅ No suspicious regions detected above threshold")

            # ── Download Buttons ──
            st.markdown("---")
            dl_col1, dl_col2, dl_col3 = st.columns(3)

            with dl_col1:
                overlay_bytes = encode_image_to_bytes(heatmap_results["overlay_bgr"])
                st.download_button(
                    "📥 Download Heatmap Overlay",
                    data=overlay_bytes,
                    file_name="heatmap_overlay.png",
                    mime="image/png",
                )

            with dl_col2:
                heatmap_bytes = encode_image_to_bytes(heatmap_results["heatmap_bgr"])
                st.download_button(
                    "📥 Download Heatmap Only",
                    data=heatmap_bytes,
                    file_name="heatmap_only.png",
                    mime="image/png",
                )

            with dl_col3:
                annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                annotated_bytes = encode_image_to_bytes(annotated_bgr)
                st.download_button(
                    "📥 Download Annotated Slide",
                    data=annotated_bytes,
                    file_name="annotated_regions.png",
                    mime="image/png",
                )

        # Close slide wrapper if open
        if hasattr(slide, 'close'):
            slide.close()

    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

else:
    # ── No File Uploaded — Instructions ──
    st.markdown("")

    inst_col1, inst_col2 = st.columns(2)

    with inst_col1:
        st.markdown("""
        ### 🚀 Getting Started

        1. **Upload** a lymph node histopathology slide image
        2. **Preview** the slide thumbnail and metadata
        3. **Adjust** analysis settings in the sidebar
        4. **Click** "Run Cancer Detection Analysis"
        5. **Review** heatmap, predictions, and suspicious regions

        ### 📁 Supported Input Formats

        | Format | Type |
        |---|---|
        | `.svs` | Aperio ScanScope |
        | `.tif` / `.tiff` | Generic tiled TIFF |
        | `.ndpi` | Hamamatsu |
        | `.jpg` / `.png` | Standard images |
        """)

    with inst_col2:
        st.markdown("""
        ### 🧠 Model Information

        | Detail | Value |
        |---|---|
        | Architecture | Vision Transformer (ViT-Base) |
        | Backbone | `vit_base_patch16_224` |
        | Patch size | 16×16 (model internal) |
        | Input resolution | 224×224 |
        | Pretraining | ImageNet-1K |
        | Task | Binary (Normal / Tumor) |

        ### 📊 Pipeline

        ```
        WSI → Tissue Detection → Patch Extraction
          → ViT Inference → Probability Grid
          → Heatmap Generation → Slide Prediction
        ```
        """)


# ═══════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════

st.markdown("""
<div class="footer">
    <strong>WSI Cancer Detection System</strong> — Vision Transformer AI Pipeline<br>
    B.Tech Final Year Project — AI-Assisted Lymph Node Pathology Analysis<br>
    Powered by ViT (timm) &amp; Streamlit
</div>
""", unsafe_allow_html=True)
