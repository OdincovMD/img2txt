"""Извлечение признаков из дерматоскопических изображений."""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import entropy
from skimage.color import deltaE_ciede2000
from skimage.measure import regionprops
from skimage.feature import local_binary_pattern

RANDOM_STATE = 42
HUE_PERIOD = 180.0
MAX_KMEANS_SAMPLES = 5000


def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected lesion component."""
    mask_bin = (mask > 0).astype(np.uint8)
    if mask_bin.sum() == 0:
        return mask_bin

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num_labels <= 1:
        return mask_bin

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8)


def _component_bbox(mask_bin: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask_bin > 0)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _shortest_signed_hue_delta(h1: float, h2: float) -> float:
    """Shortest signed difference on OpenCV hue circle [-90, 90)."""
    return float(((h1 - h2 + HUE_PERIOD / 2) % HUE_PERIOD) - HUE_PERIOD / 2)


def _circular_hue_mean(hues: np.ndarray) -> float:
    """Circular mean for OpenCV hue values in [0, 180)."""
    if len(hues) == 0:
        return 0.0
    angles = hues.astype(np.float64) * (2 * np.pi / HUE_PERIOD)
    sin_mean = np.sin(angles).mean()
    cos_mean = np.cos(angles).mean()
    if np.isclose(sin_mean, 0.0) and np.isclose(cos_mean, 0.0):
        return 0.0
    angle = np.arctan2(sin_mean, cos_mean)
    if angle < 0:
        angle += 2 * np.pi
    return float(angle * (HUE_PERIOD / (2 * np.pi)))


def _circular_hue_std(hues: np.ndarray) -> float:
    """Circular standard deviation in OpenCV hue units."""
    if len(hues) == 0:
        return 0.0
    angles = hues.astype(np.float64) * (2 * np.pi / HUE_PERIOD)
    resultant = np.hypot(np.sin(angles).mean(), np.cos(angles).mean())
    resultant = float(np.clip(resultant, 1e-12, 1.0))
    std_rad = np.sqrt(-2.0 * np.log(resultant))
    return float(std_rad * (HUE_PERIOD / (2 * np.pi)))


def _extract_skin_ring_lab(lab: np.ndarray, mask_bin: np.ndarray, ring_width: int = 5) -> np.ndarray:
    """Sample nearby skin from a narrow ring around the lesion, not from the whole background."""
    kernel = np.ones((ring_width * 2 + 1, ring_width * 2 + 1), dtype=np.uint8)
    dilated = cv2.dilate(mask_bin, kernel, iterations=1)
    ring_mask = (dilated > 0) & (mask_bin == 0)
    return lab[ring_mask]


def _dominant_hsv_clusters(lesion_pixels_hsv: np.ndarray, n_clusters: int) -> list[list[int]]:
    """Cluster HSV pixels with circular treatment of hue and return HSV-like centers."""
    if len(lesion_pixels_hsv) <= n_clusters:
        return lesion_pixels_hsv.astype(int).tolist()

    sample_pixels = lesion_pixels_hsv
    if len(sample_pixels) > MAX_KMEANS_SAMPLES:
        rng = np.random.default_rng(RANDOM_STATE)
        sample_idx = rng.choice(len(sample_pixels), size=MAX_KMEANS_SAMPLES, replace=False)
        sample_pixels = sample_pixels[sample_idx]

    hues = sample_pixels[:, 0].astype(np.float64)
    angles = hues * (2 * np.pi / HUE_PERIOD)
    features = np.column_stack([
        np.cos(angles),
        np.sin(angles),
        sample_pixels[:, 1].astype(np.float64) / 255.0,
        sample_pixels[:, 2].astype(np.float64) / 255.0,
    ])

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=5)
    labels = kmeans.fit_predict(features)

    dominant_colors = []
    for cluster_idx in range(n_clusters):
        cluster_pixels = sample_pixels[labels == cluster_idx]
        if len(cluster_pixels) == 0:
            continue
        dominant_colors.append([
            int(round(_circular_hue_mean(cluster_pixels[:, 0])) % HUE_PERIOD),
            int(round(cluster_pixels[:, 1].mean())),
            int(round(cluster_pixels[:, 2].mean())),
        ])
    return dominant_colors


def _masked_glcm(image_quantized: np.ndarray, mask_bin: np.ndarray, distances: list[int], angles: list[float], levels: int) -> np.ndarray:
    """GLCM that only counts pixel pairs fully inside the lesion mask."""
    glcm = np.zeros((levels, levels, len(distances), len(angles)), dtype=np.float64)
    rows, cols = image_quantized.shape
    mask_bool = mask_bin.astype(bool)

    for d_idx, distance in enumerate(distances):
        for a_idx, angle in enumerate(angles):
            dr = int(round(np.sin(angle) * distance))
            dc = int(round(np.cos(angle) * distance))

            src_r0 = max(0, -dr)
            src_r1 = min(rows, rows - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(cols, cols - dc)
            dst_r0 = src_r0 + dr
            dst_r1 = src_r1 + dr
            dst_c0 = src_c0 + dc
            dst_c1 = src_c1 + dc

            if src_r0 >= src_r1 or src_c0 >= src_c1:
                continue

            src_mask = mask_bool[src_r0:src_r1, src_c0:src_c1]
            dst_mask = mask_bool[dst_r0:dst_r1, dst_c0:dst_c1]
            pair_mask = src_mask & dst_mask
            if not np.any(pair_mask):
                continue

            src_vals = image_quantized[src_r0:src_r1, src_c0:src_c1][pair_mask]
            dst_vals = image_quantized[dst_r0:dst_r1, dst_c0:dst_c1][pair_mask]
            counts = np.bincount(src_vals * levels + dst_vals, minlength=levels * levels)
            counts = counts.reshape((levels, levels)).astype(np.float64)
            counts = counts + counts.T
            total = counts.sum()
            if total > 0:
                glcm[:, :, d_idx, a_idx] = counts / total
    return glcm


def _glcm_props(glcm: np.ndarray) -> tuple[float, float, float]:
    levels = glcm.shape[0]
    i_idx, j_idx = np.indices((levels, levels))

    contrast_vals = []
    homogeneity_vals = []
    energy_vals = []
    for d_idx in range(glcm.shape[2]):
        for a_idx in range(glcm.shape[3]):
            p = glcm[:, :, d_idx, a_idx]
            if p.sum() == 0:
                continue
            contrast_vals.append(float(np.sum(p * (i_idx - j_idx) ** 2)))
            homogeneity_vals.append(float(np.sum(p / (1.0 + (i_idx - j_idx) ** 2))))
            energy_vals.append(float(np.sqrt(np.sum(p ** 2))))

    if not contrast_vals:
        return 0.0, 0.0, 0.0
    return (
        float(np.mean(contrast_vals)),
        float(np.mean(homogeneity_vals)),
        float(np.mean(energy_vals)),
    )


def extract_global_color_features_with_mask(
    image: np.ndarray, mask: np.ndarray, n_clusters: int = 3
) -> dict:
    """Извлечение глобальных цветовых признаков с учётом маски."""
    features = {}
    mask_bin = _largest_component_mask(mask)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        return features

    lesion_pixels_hsv = hsv[mask_bin > 0]
    lesion_pixels_lab = lab[mask_bin > 0]
    lesion_pixels_bgr = image[mask_bin > 0]
    skin_pixels_lab = _extract_skin_ring_lab(lab, mask_bin)

    lesion_mean_hsv = np.mean(lesion_pixels_hsv[:, 1:], axis=0)
    features["mean_H_lesion"] = _circular_hue_mean(lesion_pixels_hsv[:, 0])
    features["mean_S_lesion"] = float(lesion_mean_hsv[0])
    features["mean_V_lesion"] = float(lesion_mean_hsv[1])

    lesion_std_hsv = np.std(lesion_pixels_hsv[:, 1:], axis=0)
    features["std_H_lesion"] = _circular_hue_std(lesion_pixels_hsv[:, 0])
    features["std_S_lesion"] = float(lesion_std_hsv[0])
    features["std_V_lesion"] = float(lesion_std_hsv[1])

    mean_bgr = np.mean(lesion_pixels_bgr, axis=0)
    total = np.sum(mean_bgr) + 1e-6
    balanced_bgr = mean_bgr / total
    features["color_balance_B"] = float(balanced_bgr[0])
    features["color_balance_G"] = float(balanced_bgr[1])
    features["color_balance_R"] = float(balanced_bgr[2])

    for name, channel in zip(
        ["H", "S", "V"],
        [lesion_pixels_hsv[:, 0], lesion_pixels_hsv[:, 1], lesion_pixels_hsv[:, 2]],
    ):
        hist_range = (0, 180) if name == "H" else (0, 256)
        hist, _ = np.histogram(channel, bins=16, range=hist_range, density=True)
        features[f"entropy_{name}_lesion"] = float(entropy(hist + 1e-6))

    if len(skin_pixels_lab) > 0:
        lesion_mean_lab = np.mean(lesion_pixels_lab, axis=0)
        skin_mean_lab = np.mean(skin_pixels_lab, axis=0)
        euclidean_dist = np.linalg.norm(lesion_mean_lab - skin_mean_lab)
        lesion_lab = lesion_mean_lab.reshape(1, 1, 3).astype(np.float64)
        skin_lab = skin_mean_lab.reshape(1, 1, 3).astype(np.float64)
        deltaE = float(deltaE_ciede2000(lesion_lab, skin_lab)[0, 0])
        features["color_distance_euclidean"] = float(euclidean_dist)
        features["color_distance_deltaE"] = deltaE

    dominant_colors = _dominant_hsv_clusters(lesion_pixels_hsv, n_clusters)
    if dominant_colors:
        features["dominant_colors_lesion"] = dominant_colors

    lesion_v = lesion_pixels_hsv[:, 2]
    lesion_s = lesion_pixels_hsv[:, 1]
    lesion_h = lesion_pixels_hsv[:, 0]
    features["percent_dark_pixels"] = float(np.mean(lesion_v < 50))
    features["percent_white_pixels"] = float(np.mean((lesion_v > 200) & (lesion_s < 30)))
    features["percent_red_pixels"] = float(np.mean(((lesion_h < 15) | (lesion_h > 165)) & (lesion_s > 50)))
    features["percent_blue_pixels"] = float(np.mean((lesion_h > 90) & (lesion_h < 140) & (lesion_s > 50)))
    return features


def extract_local_color_features_with_mask(image: np.ndarray, mask: np.ndarray) -> dict:
    """Извлечение локальных цветовых признаков (центр, периферия, сектора)."""
    features = {}
    mask_bin = _largest_component_mask(mask)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lesion_h = hsv[..., 0][mask_bin > 0]
    lesion_s = hsv[..., 1][mask_bin > 0]
    lesion_v = hsv[..., 2][mask_bin > 0]

    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        return features

    cy, cx = np.mean(ys), np.mean(xs)
    rr = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    r_max = rr.max()

    center_mask = rr < 0.5 * r_max
    periphery_mask = rr >= 0.5 * r_max
    if np.any(center_mask) and np.any(periphery_mask):
        center_h = lesion_h[center_mask]
        center_s = lesion_s[center_mask]
        center_v = lesion_v[center_mask]
        periph_h = lesion_h[periphery_mask]
        periph_s = lesion_s[periphery_mask]
        periph_v = lesion_v[periphery_mask]
        features["delta_H_center_periphery"] = _shortest_signed_hue_delta(
            _circular_hue_mean(center_h),
            _circular_hue_mean(periph_h),
        )
        features["delta_S_center_periphery"] = float(np.mean(center_s) - np.mean(periph_s))
        features["delta_V_center_periphery"] = float(np.mean(center_v) - np.mean(periph_v))

    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    mid_x = (x_min + x_max) // 2
    mid_y = (y_min + y_max) // 2
    left_mask = xs < mid_x
    right_mask = xs >= mid_x
    top_mask = ys < mid_y
    bottom_mask = ys >= mid_y

    if np.any(left_mask) and np.any(right_mask):
        features["delta_V_left_right"] = float(np.mean(lesion_v[left_mask]) - np.mean(lesion_v[right_mask]))
        features["delta_S_left_right"] = float(np.mean(lesion_s[left_mask]) - np.mean(lesion_s[right_mask]))
    if np.any(top_mask) and np.any(bottom_mask):
        features["delta_V_top_bottom"] = float(np.mean(lesion_v[top_mask]) - np.mean(lesion_v[bottom_mask]))
        features["delta_S_top_bottom"] = float(np.mean(lesion_s[top_mask]) - np.mean(lesion_s[bottom_mask]))

    inner_mask = rr < 0.8 * r_max
    rim_mask = (rr >= 0.8 * r_max) & (rr <= r_max)
    if np.any(inner_mask) and np.any(rim_mask):
        features["delta_V_inner_rim"] = float(np.mean(lesion_v[inner_mask]) - np.mean(lesion_v[rim_mask]))

    mean_v = np.mean(lesion_v)
    std_v = np.std(lesion_v)
    features["percent_outlier_bright_pixels"] = float(np.mean(lesion_v > mean_v + 2 * std_v))
    features["percent_outlier_dark_pixels"] = float(np.mean(lesion_v < mean_v - 2 * std_v))
    return features


def extract_shape_features(mask: np.ndarray) -> dict:
    """Извлечение признаков формы по маске."""
    features = {}
    mask_bin = _largest_component_mask(mask)
    area = np.sum(mask_bin)
    if area == 0:
        return features
    features["area"] = int(area)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = cv2.arcLength(contours[0], True) if contours else 0
    if contours:
        features["perimeter"] = float(perimeter)
    if area > 0:
        features["perimeter_area_ratio"] = float(perimeter / np.sqrt(area))
    if perimeter > 0:
        features["circularity"] = float(4 * np.pi * area / (perimeter**2))

    props_list = regionprops(mask_bin.astype(np.int32))
    if props_list:
        props = props_list[0]
        minr, minc, maxr, maxc = props.bbox
        bbox_h, bbox_w = maxr - minr, maxc - minc
        features["aspect_ratio"] = float(bbox_w / bbox_h) if bbox_h > 0 else 0.0
        features["eccentricity"] = float(props.eccentricity)
        features["solidity"] = float(props.solidity)
        features["extent"] = float(props.extent)
    return features


def _boxcount(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
        np.arange(0, Z.shape[1], k),
        axis=1,
    )
    return len(np.where((S > 0) & (S < k * k))[0])


def extract_border_features(mask: np.ndarray) -> dict:
    """Извлечение признаков границы по маске."""
    features = {}
    mask_bin = _largest_component_mask(mask)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return features
    contour = contours[0]
    perimeter = cv2.arcLength(contour, True)

    ys, xs = np.where(mask_bin > 0)
    cy, cx = np.mean(ys), np.mean(xs)
    contour_points = contour[:, 0, :]
    rr = np.sqrt((contour_points[:, 1] - cy) ** 2 + (contour_points[:, 0] - cx) ** 2)
    features["radial_variance"] = float(np.var(rr))

    hull = cv2.convexHull(contour)
    hull_perimeter = cv2.arcLength(hull, True)
    if hull_perimeter > 0:
        features["convexity"] = float(perimeter / hull_perimeter)

    Z = mask_bin.astype(bool)
    p = min(Z.shape)
    n = int(2 ** np.floor(np.log(p) / np.log(2)))
    sizes = 2 ** np.arange(int(np.log(n) / np.log(2)), 1, -1)
    counts = [_boxcount(Z, int(size)) for size in sizes]
    if len(counts) >= 2 and np.all(np.array(counts) > 0):
        coeffs = np.polyfit(np.log(1 / sizes), np.log(counts), 1)
        features["fractal_dimension"] = float(coeffs[0])
    return features


def extract_texture_features(image: np.ndarray, mask: np.ndarray) -> dict:
    """Извлечение текстурных признаков (GLCM + LBP)."""
    features = {}
    mask_bin = _largest_component_mask(mask)
    if np.sum(mask_bin) == 0:
        return features

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    y0, y1, x0, x1 = _component_bbox(mask_bin)
    gray_crop = gray[y0:y1, x0:x1]
    mask_crop = mask_bin[y0:y1, x0:x1].astype(bool)

    lesion_norm = (gray_crop / 16).astype(np.uint8)
    glcm = _masked_glcm(
        lesion_norm,
        mask_crop,
        distances=[1, 2, 4],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=16,
    )
    contrast, homogeneity, energy = _glcm_props(glcm)
    p = glcm.flatten()
    features["glcm_contrast"] = float(contrast)
    features["glcm_homogeneity"] = float(homogeneity)
    features["glcm_energy"] = float(energy)
    features["glcm_entropy"] = float(entropy(p + 1e-6))

    lbp = local_binary_pattern(gray_crop, P=8, R=1, method="uniform")
    valid_lbp_mask = cv2.erode(mask_crop.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
    lesion_lbp = lbp[valid_lbp_mask]
    if lesion_lbp.size == 0:
        lesion_lbp = lbp[mask_crop]

    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lesion_lbp, bins=n_bins, range=(0, n_bins), density=True)
    features["lbp_uniformity"] = float(np.max(hist))
    features["lbp_entropy"] = float(entropy(hist + 1e-6))
    features["lbp_mean"] = float(np.mean(lesion_lbp))
    features["lbp_std"] = float(np.std(lesion_lbp))
    features["lbp_median"] = float(np.median(lesion_lbp))
    return features

