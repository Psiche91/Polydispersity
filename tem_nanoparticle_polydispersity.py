#!/usr/bin/env python3
"""
tem_nanoparticle_polydispersity.py

Misura automaticamente il diametro di nanoparticelle sferiche in immagini TEM
Bright Field (particelle impaccate/anche a contatto tra loro) e calcola le
statistiche di polidispersita' (media, deviazione standard, CV%, PDI).

Pipeline:
  1. Calibrazione px -> nm tramite la scale bar nera presente nell'immagine.
  2. Sogliatura (Otsu) per separare le particelle (scure) dal background (chiaro).
  3. Segmentazione watershed (basata sulla distance transform) per separare
     le particelle che si toccano tra loro.
  4. Misura di area/diametro equivalente/circolarita' per ogni particella,
     con filtro dei frammenti troppo piccoli o poco circolari (rumore).
  5. Calcolo statistiche di polidispersita' + istogramma + overlay di verifica.

Dipendenze:
    pip install opencv-python-headless numpy scipy scikit-image matplotlib

Esempio d'uso da riga di comando:
    python tem_nanoparticle_polydispersity.py immagine.jpg \
        --scalebar-nm 200 \
        --scalebar-region 0.85 1.0 0.0 0.35 \
        --min-diam-px 25 \
        --outdir risultati

Esempio d'uso come libreria:
    from tem_nanoparticle_polydispersity import analyze_image

    result = analyze_image("immagine.jpg", scalebar_nm=200)
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
# 1. Calibrazione della scale bar
# --------------------------------------------------------------------------
def calibrate_scalebar(gray_img, scalebar_nm=200.0, region=(0.85, 1.0, 0.0, 0.35),
                        dark_thresh=100, min_bar_px=30):
    """
    Trova la lunghezza in pixel della scale bar (barra nera solida e
    orizzontale) all'interno di una sotto-regione dell'immagine e calcola il
    fattore di conversione nm/pixel.

    Parametri
    ---------
    gray_img : np.ndarray (2D, uint8)
        Immagine in scala di grigi.
    scalebar_nm : float
        Lunghezza reale della scale bar in nm (es. 200).
    region : tuple(float, float, float, float)
        Frazioni (row_start, row_end, col_start, col_end) della regione
        dell'immagine in cui cercare la scale bar. Default: angolo in basso
        a sinistra.
    dark_thresh : int
        Soglia di intensita' sotto la quale un pixel e' considerato "nero"
        (parte della scale bar).
    min_bar_px : int
        Numero minimo di pixel scuri consecutivi in una riga perche' venga
        considerata come riga della scale bar (evita di confondere il testo
        "200 nm" con la barra).

    Ritorna
    -------
    dict con:
        nm_per_px : float
        bar_length_px : float
        bar_bbox : (row, col_start, col_end) nelle coordinate dell'immagine originale
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
        # la barra e' un segmento continuo: prendo min/max e verifico che
        # la maggior parte dei pixel nel range sia effettivamente scura
        # (per escludere righe di testo, che sono piu' "sparse")
        span = cols.max() - cols.min() + 1
        if span < min_bar_px:
            continue
        fill_ratio = len(cols) / span
        if fill_ratio > 0.9 and span > best_len:
            best_row, best_start, best_end, best_len = row, cols.min(), cols.max(), span

    if best_row is None:
        raise RuntimeError(
            "Scale bar non trovata automaticamente: prova a cambiare 'region' "
            "o 'dark_thresh'."
        )

    bar_length_px = best_end - best_start + 1
    nm_per_px = scalebar_nm / bar_length_px

    return {
        "nm_per_px": nm_per_px,
        "bar_length_px": bar_length_px,
        "bar_bbox": (best_row + r0, best_start + c0, best_end + c0),
    }


# --------------------------------------------------------------------------
# 2-3. Segmentazione (Otsu) + separazione delle particelle a contatto (watershed)
# --------------------------------------------------------------------------
def segment_and_split_particles(gray_img, exclude_region=None, min_peak_distance=15):
    """
    Segmenta le particelle scure su sfondo chiaro e separa quelle a contatto
    tramite watershed basato sulla distance transform.

    Parametri
    ---------
    gray_img : np.ndarray (2D, uint8)
    exclude_region : tuple(float, float, float, float) opzionale
        Regione (row_start, row_end, col_start, col_end), in frazioni, da
        escludere dalla segmentazione (es. dove si trova la scale bar/testo),
        impostata a colore di sfondo prima della sogliatura.
    min_peak_distance : int
        Distanza minima (in pixel) tra i massimi locali della distance
        transform, usata come marker per il watershed. Va scelta in base al
        raggio approssimativo delle particelle (piu' piccola se le
        particelle sono piccole/ravvicinate).

    Ritorna
    -------
    labels_ws : np.ndarray (2D, int) - etichette watershed (0 = background)
    mask : np.ndarray (2D, bool) - maschera binaria delle particelle
    """
    h, w = gray_img.shape
    work = gray_img.copy()

    if exclude_region is not None:
        r0, r1, c0, c1 = exclude_region
        r0, r1 = int(h * r0), int(h * r1)
        c0, c1 = int(w * c0), int(w * c1)
        work[r0:r1, c0:c1] = 255  # forza a "sfondo chiaro"

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
# 4. Misura delle particelle + filtro rumore
# --------------------------------------------------------------------------
def measure_particles(labels_ws, nm_per_px, min_diam_px=25, min_circularity=0.5):
    """
    Calcola area, diametro equivalente e circolarita' di ogni regione
    watershed e filtra le regioni troppo piccole o poco circolari (in
    genere frammenti di rumore o artefatti ai bordi).

    Ritorna
    -------
    list di dict, uno per particella accettata, con chiavi:
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
# Statistiche di polidispersita'
# --------------------------------------------------------------------------
def polydispersity_stats(diam_nm):
    """
    Calcola le statistiche di polidispersita' a partire da un array di
    diametri (in nm).

    PDI qui e' definito come (sigma/media)^2, la stessa convenzione usata
    tipicamente in DLS. E' una metrica adimensionale: valori piu' bassi
    indicano un campione piu' monodisperso (es. PDI < 0.05 - 0.1 in TEM).
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
# Visualizzazione
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


def plot_histogram(diam_nm, stats, out_path, material="nanoparticelle"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    plt.hist(diam_nm, bins=30, color="#4C72B0", edgecolor="white")
    plt.axvline(stats["mean_nm"], color="crimson", linestyle="--", linewidth=1.5,
                label=f'Media = {stats["mean_nm"]:.2f} nm')
    plt.xlabel("Diametro particella (nm)")
    plt.ylabel("Conteggio")
    plt.title(
        f'Distribuzione dimensionale {material} (N={stats["n"]})\n'
        f'Dev.std = {stats["std_nm"]:.2f} nm, CV = {stats["cv_percent"]:.1f}%, '
        f'PDI = {stats["pdi"]:.4f}'
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# Pipeline completa
# --------------------------------------------------------------------------
def analyze_image(image_path, scalebar_nm=200.0,
                   scalebar_region=(0.85, 1.0, 0.0, 0.35),
                   min_peak_distance=15, min_diam_px=25, min_circularity=0.5,
                   material="nanoparticelle", outdir=None):
    """
    Esegue l'intera pipeline su un'immagine TEM e ritorna un dizionario con
    calibrazione, lista particelle e statistiche. Se outdir e' specificato,
    salva anche overlay.png e histogram.png.
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Impossibile aprire l'immagine: {image_path}")

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
        description="Misura la polidispersita' di nanoparticelle in immagini TEM Bright Field."
    )
    ap.add_argument("image", help="Percorso dell'immagine TEM (es. jpg/png/tif)")
    ap.add_argument("--scalebar-nm", type=float, default=200.0,
                     help="Lunghezza reale della scale bar in nm (default: 200)")
    ap.add_argument("--scalebar-region", type=float, nargs=4,
                     default=(0.85, 1.0, 0.0, 0.35),
                     metavar=("ROW_START", "ROW_END", "COL_START", "COL_END"),
                     help="Regione (frazioni 0-1) dove cercare la scale bar. "
                          "Default: angolo in basso a sinistra.")
    ap.add_argument("--min-peak-distance", type=int, default=15,
                     help="Distanza minima tra centri particella per il watershed (px)")
    ap.add_argument("--min-diam-px", type=float, default=25,
                     help="Diametro minimo (px) per accettare una particella (filtro rumore)")
    ap.add_argument("--min-circularity", type=float, default=0.5,
                     help="Circolarita' minima (4*pi*area/perimetro^2) per accettare una particella")
    ap.add_argument("--material", default="nanoparticelle",
                     help="Nome del materiale, usato solo nel titolo del grafico (es. 'nanoparticelle di Sn')")
    ap.add_argument("--outdir", default="risultati_TEM",
                     help="Cartella di output per overlay.png e histogram.png")
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
    print(f"Calibrazione: {calib['bar_length_px']:.1f} px = {args.scalebar_nm:.0f} nm "
          f"-> {calib['nm_per_px']:.4f} nm/px")
    print(f"Particelle rilevate (dopo filtro rumore): {stats['n']}")
    print(f"Diametro medio:     {stats['mean_nm']:.2f} nm")
    print(f"Diametro mediano:   {stats['median_nm']:.2f} nm")
    print(f"Deviazione std:     {stats['std_nm']:.2f} nm")
    print(f"CV (RSD):           {stats['cv_percent']:.2f} %")
    print(f"PDI (sigma/media)^2:{stats['pdi']:.4f}")
    print(f"Range:              {stats['min_nm']:.2f} - {stats['max_nm']:.2f} nm")
    print("=" * 60)
    print(f"Risultati grafici salvati in: {args.outdir}/overlay.png, {args.outdir}/histogram.png")


if __name__ == "__main__":
    main()
