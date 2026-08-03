#!/usr/bin/env python3
"""
tem_nanoparticle_polydispersity.py

Automatically measures the diameter of spherical nanoparticles in Bright
Field TEM images (packed particles, possibly touching each other) and
computes polydispersity statistics (mean, standard deviation, CV%, PDI).

Pipeline:
  1. Pixel -> nm calibration via the black scale bar present in the image.
  2. Thresholding (Otsu) to separate particles (dark) from background (light).
  3. Watershed segmentation (based on the distance transform) to separate
     particles that are touching each other.
  4. Measurement of area/equivalent diameter/circularity for each particle,
     filtering out fragments that are too small or not circular enough
     (noise).
  5. Computation of polydispersity statistics + histogram + verification
     overlay.

Dependencies:
    pip install opencv-python-headless numpy scipy scikit-image matplotlib

Command-line usage example:
    python tem_nanoparticle_polydispersity.py image.jpg \
        --scalebar-nm 200 \
        --scalebar-region 0.85 1.0 0.0 0.35 \
        --min-diam-px 25 \
        --outdir results

Library usage example:
    from tem_nanoparticle_polydispersity import analyze_image

    result = analyze_image("image.jpg", scalebar_nm=200)
    print(result["stats"])
"""

import argparse
import os

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops


# --------------------------------------------------------------------------
# 1. Scale bar calibration
# --------------------------------------------------------------------------
def calibrate_scalebar(gray_img, scalebar_nm=200.0, region=(0.85, 1.0, 0.0, 0.35),
                        dark_thresh=100, min_bar_px=30):
    """
    Finds the length in pixels of the scale bar (solid, horizontal black
    bar) within a sub-region of the image and computes the nm/pixel
    conversion factor.

    Parameters
    ----------
    gray_img : np.ndarray (2D, uint8)
        Grayscale image.
    scalebar_nm : float
        Real length of the scale bar in nm (e.g. 200).
    region : tuple(float, float, float, float)
        Fractions (row_start, row_end, col_start, col_end) of the image
        region in which to search for the scale bar. Default: bottom-left
        corner.
    dark_thresh : int
        Intensity threshold below which a pixel is considered "black"
        (part of the scale bar).
    min_bar_px : int
        Minimum number of consecutive dark pixels in a row for it to be
        considered a scale bar row (avoids confusing the "200 nm" text
        with the bar itself).

    Returns
    -------
    dict with:
        nm_per_px : float
        bar_length_px : float
        bar_bbox : (row, col_start, col_end) in the original image coordinates
    """
    h, w = gray_img.shape
    r0, r1, c0, c1 = region
    r0, r1 = int(h * r0), int(h * r1)
    c0, c1 = int(w * c0), int(w * c1)
    crop = gray_img[r0:r1, c0:c1]

    dark_mask = crop < dark_thresh

    best_row, best_start, best_end, best_len = None, None, None, 0
    for row in range(dark_mask.shape[0]):
        cols = np.where(dark_mask[row])[0]
        if len(cols) < min_bar_px:
            continue
        # the bar is a continuous segment: take min/max and check that
        # most pixels in that range are actually dark
        # (to exclude text rows, which are more "sparse")
        span = cols.max() - cols.min() + 1
        if span < min_bar_px:
            continue
        fill_ratio = len(cols) / span
        if fill_ratio > 0.9 and span > best_len:
            best_row, best_start, best_end, best_len = row, cols.min(), cols.max(), span

    if best_row is None:
        raise RuntimeError(
            "Scale bar not found automatically: try changing 'region' "
            "or 'dark_thresh'."
        )

    bar_length_px = best_end - best_start + 1
    nm_per_px = scalebar_nm / bar_length_px

    return {
        "nm_per_px": nm_per_px,
        "bar_length_px": bar_length_px,
        "bar_bbox": (best_row + r0, best_start + c0, best_end + c0),
    }


# --------------------------------------------------------------------------
# 2-3. Segmentation (Otsu) + splitting touching particles (watershed)
# --------------------------------------------------------------------------
def segment_and_split_particles(gray_img, exclude_region=None, min_peak_distance=15):
    """
    Segments dark particles on a light background and splits touching
    particles using watershed based on the distance transform.

    Parameters
    ----------
    gray_img : np.ndarray (2D, uint8)
    exclude_region : tuple(float, float, float, float), optional
        Region (row_start, row_end, col_start, col_end), in fractions, to
        exclude from segmentation (e.g. where the scale bar/text is
        located), set to background color before thresholding.
    min_peak_distance : int
        Minimum distance (in pixels) between local maxima of the distance
        transform, used as markers for the watershed. Should be chosen
        based on the approximate particle radius (smaller if particles are
        small/closely packed).

    Returns
    -------
    labels_ws : np.ndarray (2D, int) - watershed labels (0 = background)
    mask : np.ndarray (2D, bool) - binary mask of the particles
    """
    h, w = gray_img.shape
    work = gray_img.copy()

    if exclude_region is not None:
        r0, r1, c0, c1 = exclude_region
        r0, r1 = int(h * r0), int(h * r1)
        c0, c1 = int(w * c0), int(w * c1)
        work[r0:r1, c0:c1] = 255  # force to "light background"

    blur = cv2.GaussianBlur(work, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    th_clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th_clean = cv2.morphologyEx(th_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = th_clean > 0

    dist = ndi.distance_transform_edt(mask)
    coords = peak_local_max(dist, min_distance=min_peak_distance, labels=mask,
                             exclude_border=False)
    markers_mask = np.zeros(dist.shape, dtype=bool)
    markers_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(markers_mask)

    labels_ws = watershed(-dist, markers, mask=mask)
    return labels_ws, mask


# --------------------------------------------------------------------------
# 4. Particle measurement + noise filtering
# --------------------------------------------------------------------------
def measure_particles(labels_ws, nm_per_px, min_diam_px=25, min_circularity=0.5):
    """
    Computes area, equivalent diameter and circularity for each watershed
    region and filters out regions that are too small or not circular
    enough (typically noise fragments or edge artifacts).

    Returns
    -------
    list of dict, one per accepted particle, with keys:
        label, area_px, eq_diam_px, eq_diam_nm, circularity, solidity, centroid
    """
    props = regionprops(labels_ws)
    particles = []
    for p in props:
        area = p.area
        perim = p.perimeter if p.perimeter > 0 else 1
        circularity = 4 * np.pi * area / (perim ** 2)
        eq_diam_px = 2 * np.sqrt(area / np.pi)

        if eq_diam_px < min_diam_px:
            continue
        if circularity < min_circularity:
            continue

        particles.append({
            "label": p.label,
            "area_px": area,
            "eq_diam_px": eq_diam_px,
            "eq_diam_nm": eq_diam_px * nm_per_px,
            "circularity": circularity,
            "solidity": p.solidity,
            "centroid": p.centroid,  # (row, col)
        })
    return particles


# --------------------------------------------------------------------------
# Polydispersity statistics
# --------------------------------------------------------------------------
def polydispersity_stats(diam_nm):
    """
    Computes polydispersity statistics from an array of diameters (in nm).

    PDI here is defined as (sigma/mean)^2, the same convention typically
    used in DLS. It is a dimensionless metric: lower values indicate a
    more monodisperse sample (e.g. PDI < 0.05 - 0.1 in TEM).
    """
    diam_nm = np.asarray(diam_nm)
    mean_d = diam_nm.mean()
    std_d = diam_nm.std(ddof=1)
    return {
        "n": len(diam_nm),
        "mean_nm": mean_d,
        "median_nm": float(np.median(diam_nm)),
        "std_nm": std_d,
        "cv_percent": std_d / mean_d * 100,
        "pdi": (std_d / mean_d) ** 2,
        "min_nm": float(diam_nm.min()),
        "max_nm": float(diam_nm.max()),
    }


# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------
def draw_overlay(color_img, particles, out_path):
    overlay = color_img.copy()
    for p in particles:
        cy, cx = p["centroid"]
        r = p["eq_diam_px"] / 2
        cv2.circle(overlay, (int(round(cx)), int(round(cy))), int(round(r)),
                   (0, 0, 255), 2)
        cv2.circle(overlay, (int(round(cx)), int(round(cy))), 1, (0, 255, 0), 2)
    cv2.imwrite(out_path, overlay)


def plot_histogram(diam_nm, stats, out_path, material="nanoparticles"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    plt.hist(diam_nm, bins=30, color="#4C72B0", edgecolor="white")
    plt.axvline(stats["mean_nm"], color="crimson", linestyle="--", linewidth=1.5,
                label=f'Mean = {stats["mean_nm"]:.2f} nm')
    plt.xlabel("Particle diameter (nm)")
    plt.ylabel("Count")
    plt.title(
        f'Size distribution {material} (N={stats["n"]})\n'
        f'Std dev = {stats["std_nm"]:.2f} nm, CV = {stats["cv_percent"]:.1f}%, '
        f'PDI = {stats["pdi"]:.4f}'
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------
def analyze_image(image_path, scalebar_nm=200.0,
                   scalebar_region=(0.85, 1.0, 0.0, 0.35),
                   min_peak_distance=15, min_diam_px=25, min_circularity=0.5,
                   material="nanoparticles", outdir=None):
    """
    Runs the entire pipeline on a TEM image and returns a dictionary with
    calibration, particle list and statistics. If outdir is specified,
    also saves overlay.png and histogram.png.
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    calib = calibrate_scalebar(gray, scalebar_nm=scalebar_nm, region=scalebar_region)

    labels_ws, mask = segment_and_split_particles(
        gray, exclude_region=scalebar_region, min_peak_distance=min_peak_distance
    )

    particles = measure_particles(
        labels_ws, nm_per_px=calib["nm_per_px"],
        min_diam_px=min_diam_px, min_circularity=min_circularity,
    )
    diam_nm = np.array([p["eq_diam_nm"] for p in particles])
    stats = polydispersity_stats(diam_nm)

    result = {"calibration": calib, "particles": particles, "stats": stats}

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        color_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        draw_overlay(color_img, particles, os.path.join(outdir, "overlay.png"))
        plot_histogram(diam_nm, stats, os.path.join(outdir, "histogram.png"),
                        material=material)

    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_args():
    ap = argparse.ArgumentParser(
        description="Measures the polydispersity of nanoparticles in Bright Field TEM images."
    )
    ap.add_argument("image", help="Path to the TEM image (e.g. jpg/png/tif)")
    ap.add_argument("--scalebar-nm", type=float, default=200.0,
                     help="Real length of the scale bar in nm (default: 200)")
    ap.add_argument("--scalebar-region", type=float, nargs=4,
                     default=(0.85, 1.0, 0.0, 0.35),
                     metavar=("ROW_START", "ROW_END", "COL_START", "COL_END"),
                     help="Region (fractions 0-1) where to search for the scale bar. "
                          "Default: bottom-left corner.")
    ap.add_argument("--min-peak-distance", type=int, default=15,
                     help="Minimum distance between particle centers for watershed (px)")
    ap.add_argument("--min-diam-px", type=float, default=25,
                     help="Minimum diameter (px) to accept a particle (noise filter)")
    ap.add_argument("--min-circularity", type=float, default=0.5,
                     help="Minimum circularity (4*pi*area/perimeter^2) to accept a particle")
    ap.add_argument("--material", default="nanoparticles",
                     help="Material name, used only in the plot title (e.g. 'Sn nanoparticles')")
    ap.add_argument("--outdir", default="results_TEM",
                     help="Output folder for overlay.png and histogram.png")
    return ap.parse_args()


def main():
    args = _parse_args()
    result = analyze_image(
        args.image,
        scalebar_nm=args.scalebar_nm,
        scalebar_region=tuple(args.scalebar_region),
        min_peak_distance=args.min_peak_distance,
        min_diam_px=args.min_diam_px,
        min_circularity=args.min_circularity,
        material=args.material,
        outdir=args.outdir,
    )

    calib = result["calibration"]
    stats = result["stats"]

    print("=" * 60)
    print(f"Calibration: {calib['bar_length_px']:.1f} px = {args.scalebar_nm:.0f} nm "
          f"-> {calib['nm_per_px']:.4f} nm/px")
    print(f"Particles detected (after noise filter): {stats['n']}")
    print(f"Mean diameter:      {stats['mean_nm']:.2f} nm")
    print(f"Median diameter:    {stats['median_nm']:.2f} nm")
    print(f"Std deviation:      {stats['std_nm']:.2f} nm")
    print(f"CV (RSD):           {stats['cv_percent']:.2f} %")
    print(f"PDI (sigma/mean)^2: {stats['pdi']:.4f}")
    print(f"Range:              {stats['min_nm']:.2f} - {stats['max_nm']:.2f} nm")
    print("=" * 60)
    print(f"Plot results saved in: {args.outdir}/overlay.png, {args.outdir}/histogram.png")


if __name__ == "__main__":
    main()
