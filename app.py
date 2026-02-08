"""
Streamlit-based dataset preprocessing tool for plant disease ROI highlighting and YOLO-ready
annotation generation.

This application lets a user load raw plant images, manually annotate the diseased
regions using a simple drawing canvas, preview the effect of background processing
(Gaussian blur or grayscale), and export the processed image, binary mask, YOLO
labels and metadata. The tool keeps the original image resolution intact and
stores all output files in a strict directory structure.

Functional highlights:

* Dataset loader with sidebar navigation and progress indicator.
* Free‑draw annotation canvas overlaid on each image. A toolbar allows undo,
  redo and reset operations. A brush size slider lets users control the stroke
  width.
* Real‑time preview of the processed image, where the annotated diseased area
  remains sharp and the background is either blurred or converted to grayscale.
* Save & Next workflow: upon saving, the processed image, binary mask, YOLO
  label file and a metadata JSON file are written to disk. The app then
  automatically advances to the next image.

The application relies on the ``streamlit-drawable-canvas-fix`` component for
freehand drawing. According to its documentation, the canvas supports free
drawing as well as other primitives, and it can overlay a background image
without including it in the returned drawing data【947088014688088†L94-L102】.  This
property allows us to recover a clean mask from the drawing overlay by
examining its alpha channel. The canvas also provides undo/redo/delete
operations out of the box【947088014688088†L94-L102】, which serve as a basic
eraser for the user.

The YOLO label format is based on the Ultralytics specification: each line in
the ``.txt`` file contains ``class x_center y_center width height`` with
coordinates normalised by the image width and height【545789746564443†L271-L276】.  All
boxes are derived from the connected components of the binary mask.

Author: ChatGPT (Senior Computer Vision Engineer)
"""

import os
import json
import datetime
from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas  # type: ignore


def ensure_directories(base_dir: Path) -> None:
    """Ensure that all required directories exist under the project root.

    The directory structure is strictly defined as:
    project_root/
        raw_images/
        processed_images/
        masks/
        labels/
        metadata/

    Parameters
    ----------
    base_dir : Path
        Root directory of the project. Relative directories will be created
        underneath.
    """
    required = [
        base_dir / "raw_images",
        base_dir / "processed_images",
        base_dir / "masks",
        base_dir / "labels",
        base_dir / "metadata",
    ]
    for d in required:
        d.mkdir(parents=True, exist_ok=True)


def list_images(raw_dir: Path) -> list[str]:
    """Return a sorted list of image filenames in ``raw_dir``.

    Only files with extensions typically associated with images are returned.
    Sorting ensures deterministic ordering.

    Parameters
    ----------
    raw_dir : Path
        Directory containing the raw images.

    Returns
    -------
    list[str]
        Sorted list of image filenames.
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(
        [f.name for f in raw_dir.iterdir() if f.is_file() and f.suffix.lower() in exts]
    )


def load_image(path: Path) -> np.ndarray:
    """Load an image from disk into a BGR NumPy array.

    Parameters
    ----------
    path : Path
        Path to the image file.

    Returns
    -------
    np.ndarray
        Image as a BGR array with shape (H, W, 3).
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def create_mask_from_canvas(canvas_data: np.ndarray) -> np.ndarray:
    """Convert the canvas RGBA array into a binary mask.

    The returned ``image_data`` from ``st_canvas`` contains only the drawing
    overlay; the background image is not included【947088014688088†L187-L190】.  The
    alpha channel therefore indicates drawn pixels. A non‑zero alpha value
    corresponds to user‑annotated disease regions.

    Parameters
    ----------
    canvas_data : np.ndarray
        Array with shape (H, W, 4) where the last channel is alpha.

    Returns
    -------
    np.ndarray
        Binary mask with values 0 (background) or 255 (disease region).
    """
    if canvas_data.ndim != 3 or canvas_data.shape[2] != 4:
        raise ValueError("Canvas data must be an RGBA image (H, W, 4).")
    alpha = canvas_data[:, :, 3]
    mask = (alpha > 0).astype(np.uint8) * 255
    return mask


def apply_background_effect(
    original_bgr: np.ndarray,
    mask: np.ndarray,
    effect: str,
    blur_strength: int = 5,
) -> np.ndarray:
    """Apply a background effect (blur or grayscale) while preserving the mask area.

    Parameters
    ----------
    original_bgr : np.ndarray
        Original image in BGR format.
    mask : np.ndarray
        Binary mask where disease pixels are 255 and background is 0.
    effect : str
        Either ``'blur'`` or ``'grayscale'``.
    blur_strength : int, optional
        Kernel size for Gaussian blur. Must be an odd integer. Defaults to 5.

    Returns
    -------
    np.ndarray
        Processed BGR image with the background modified and disease area
        untouched.
    """
    # Ensure mask is binary 0/255
    mask_bool = mask > 0
    h, w = original_bgr.shape[:2]
    if effect == "blur":
        k = max(1, blur_strength)
        # Kernel size must be odd; if even, increment by 1
        if k % 2 == 0:
            k += 1
        blurred = cv2.GaussianBlur(original_bgr, (k, k), 0)
        processed = original_bgr.copy()
        processed[~mask_bool] = blurred[~mask_bool]
        return processed
    elif effect == "grayscale":
        gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        processed = original_bgr.copy()
        processed[~mask_bool] = gray_bgr[~mask_bool]
        return processed
    else:
        raise ValueError(f"Unknown effect: {effect}")


def generate_yolo_labels(mask: np.ndarray) -> list[str]:
    """Compute YOLO formatted labels from a binary mask.

    Each connected component in the mask is converted into a bounding box. The
    bounding box coordinates are normalised by the image dimensions according
    to the Ultralytics YOLO format specification【545789746564443†L271-L276】.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask with 0/255 values.

    Returns
    -------
    list[str]
        List of YOLO annotation lines. Each line has the form
        ``"0 x_center y_center width height"``.
    """
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labels = []
    for cnt in contours:
        if cv2.contourArea(cnt) == 0:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        w_norm = bw / w
        h_norm = bh / h
        labels.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
    return labels


def save_outputs(
    filename: str,
    processed_bgr: np.ndarray,
    mask: np.ndarray,
    labels: list[str],
    effect: str,
    blur_strength: int,
    output_dirs: dict[str, Path],
) -> None:
    """Write processed image, mask, label file and metadata to disk.

    Parameters
    ----------
    filename : str
        Original image filename (with extension).
    processed_bgr : np.ndarray
        Processed image in BGR format.
    mask : np.ndarray
        Binary mask (0/255) of the same size as original.
    labels : list[str]
        YOLO annotation lines.
    effect : str
        Background effect used ('blur' or 'grayscale').
    blur_strength : int
        Blur kernel size used. If effect is 'grayscale', this will be 0.
    output_dirs : dict[str, Path]
        Mapping of output categories to their directories.
    """
    stem = Path(filename).stem
    # Save processed image
    cv2.imwrite(str(output_dirs["processed_images"] / filename), processed_bgr)
    # Save mask as PNG (white=255, black=0)
    cv2.imwrite(str(output_dirs["masks"] / f"{stem}.png"), mask)
    # Save YOLO labels
    label_path = output_dirs["labels"] / f"{stem}.txt"
    if labels:
        with open(label_path, "w") as f:
            f.write("\n".join(labels))
    else:
        # If no labels, ensure any existing file is removed to indicate no objects
        if label_path.exists():
            label_path.unlink()
    # Save metadata
    metadata = {
        "blur_type": effect,
        "blur_strength": blur_strength if effect == "blur" else 0,
        "image_resolution": {
            "width": processed_bgr.shape[1],
            "height": processed_bgr.shape[0],
        },
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with open(output_dirs["metadata"] / f"{stem}.json", "w") as f:
        json.dump(metadata, f, indent=4)


def app():
    """Entry point for the Streamlit application."""
    st.set_page_config(
        page_title="Disease ROI Highlighting & YOLO Dataset Preparation",
        layout="wide",
    )
    st.title("Plant Disease ROI Highlighting & YOLO Dataset Preparation Tool")
    base_dir = Path(__file__).resolve().parent
    ensure_directories(base_dir)
    dirs = {
        "raw_images": base_dir / "raw_images",
        "processed_images": base_dir / "processed_images",
        "masks": base_dir / "masks",
        "labels": base_dir / "labels",
        "metadata": base_dir / "metadata",
    }
    image_files = list_images(dirs["raw_images"])
    if not image_files:
        st.warning(
            "No images found in 'raw_images/'. Please add images to annotate and restart the app."
        )
        return
    # Determine processed flags based on metadata existence
    processed_flags = {
        fname: (dirs["metadata"] / f"{Path(fname).stem}.json").exists()
        for fname in image_files
    }
    total_count = len(image_files)
    processed_count = sum(processed_flags.values())
    # Initialize session state for navigation index
    if "img_idx" not in st.session_state:
        st.session_state["img_idx"] = 0
    # Sidebar UI
    with st.sidebar:
        st.header("Dataset Loader")
        st.write(f"Progress: {processed_count} / {total_count} images annotated")
        # List of images with status indicator
        # Use radio buttons to allow selection of any image
        display_names = []
        for fname in image_files:
            prefix = "✅" if processed_flags[fname] else "⬜"
            display_names.append(f"{prefix} {fname}")
        selected = st.radio("Select image", display_names, index=st.session_state["img_idx"])
        # Update index based on selection
        st.session_state["img_idx"] = display_names.index(selected)
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Previous"):
                st.session_state["img_idx"] = (st.session_state["img_idx"] - 1) % total_count
        with col2:
            if st.button("Next"):
                st.session_state["img_idx"] = (st.session_state["img_idx"] + 1) % total_count
        # Background effect selection
        st.markdown("---")
        st.subheader("Background Effect")
        effect_choice = st.radio(
            "Choose effect", ["Gaussian Blur", "Grayscale"], key="effect_choice"
        )
        if effect_choice == "Gaussian Blur":
            blur_strength = st.slider(
                "Blur kernel size (odd)", min_value=3, max_value=51, value=15, step=2
            )
        else:
            blur_strength = 0
        # Brush size
        st.markdown("---")
        stroke_width = st.slider("Brush size (pixels)", min_value=1, max_value=100, value=30)
        # Save button
        if st.button("Save & Next"):
            st.session_state["save_trigger"] = True
        else:
            st.session_state["save_trigger"] = False

    # Main panel
    current_file = image_files[st.session_state["img_idx"]]
    img_path = dirs["raw_images"] / current_file
    original_bgr = load_image(img_path)
    height, width = original_bgr.shape[:2]
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    # Annotation canvas
    st.subheader(f"Annotating: {current_file} ({st.session_state['img_idx']+1}/{total_count})")
    st.write("**Draw over the diseased regions using the brush. Use the toolbar to undo/redo or reset.**")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",  # semi‑transparent red fill if polygons are used
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",  # white strokes for disease annotation
        background_image=Image.fromarray(original_rgb),
        update_streamlit=True,
        height=height,
        width=width,
        drawing_mode="freedraw",
        display_toolbar=True,
        key=f"canvas_{current_file}",
    )
    # Prepare preview and output if drawing exists
    mask = None
    processed_rgb = None
    yolo_labels = []
    if canvas_result.image_data is not None:
        # Convert to mask
        canvas_data = np.array(canvas_result.image_data, dtype=np.uint8)
        mask = create_mask_from_canvas(canvas_data)
        # Apply background effect
        effect_key = "blur" if effect_choice == "Gaussian Blur" else "grayscale"
        processed_bgr = apply_background_effect(
            original_bgr,
            mask,
            effect=effect_key,
            blur_strength=blur_strength,
        )
        processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        # Compute YOLO labels for preview (not saved until Save & Next)
        yolo_labels = generate_yolo_labels(mask)
        st.image(processed_rgb, caption="Processed Preview", use_column_width=True)
    else:
        st.info("Draw on the image to begin annotation.")
    # Save & Next logic
    if st.session_state.get("save_trigger", False):
        if mask is None or processed_rgb is None:
            st.warning("Nothing to save. Please annotate the image first.")
        else:
            # Save outputs
            save_outputs(
                filename=current_file,
                processed_bgr=processed_bgr,
                mask=mask,
                labels=yolo_labels,
                effect=effect_key,
                blur_strength=blur_strength,
                output_dirs=dirs,
            )
            st.success(f"Saved {current_file} and generated labels.")
            # Move to next image
            next_index = (st.session_state["img_idx"] + 1) % total_count
            st.session_state["img_idx"] = next_index
            # Trigger a rerun to refresh the UI. Streamlit renamed
            # experimental_rerun() to rerun() in recent releases. Use the
            # new API if available for compatibility with older versions.
            try:
                # Streamlit >= 1.25.0 provides st.rerun()
                rerun_func = getattr(st, "rerun")
            except AttributeError:
                # Fall back to experimental API for earlier versions
                rerun_func = getattr(st, "experimental_rerun")
            rerun_func()

if __name__ == "__main__":
    app()
