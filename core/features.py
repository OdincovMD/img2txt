"""Извлечение признаков из дерматоскопических изображений."""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import entropy
from skimage.color import deltaE_ciede2000
from skimage.measure import regionprops
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

RANDOM_STATE = 42


def extract_global_color_features_with_mask(
    image: np.ndarray, mask: np.ndarray, n_clusters: int = 3
) -> dict:
    """Извлечение глобальных цветовых признаков с учётом маски."""
    features = {}
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return features

    lesion_pixels_hsv = hsv[mask > 0]
    lesion_pixels_lab = lab[mask > 0]
    lesion_pixels_bgr = image[mask > 0]
    skin_pixels_lab = lab[mask == 0]

    lesion_mean_hsv = np.mean(lesion_pixels_hsv, axis=0)
    features["mean_H_lesion"] = float(lesion_mean_hsv[0])
    features["mean_S_lesion"] = float(lesion_mean_hsv[1])
    features["mean_V_lesion"] = float(lesion_mean_hsv[2])

    lesion_std_hsv = np.std(lesion_pixels_hsv, axis=0)
    features["std_H_lesion"] = float(lesion_std_hsv[0])
    features["std_S_lesion"] = float(lesion_std_hsv[1])
    features["std_V_lesion"] = float(lesion_std_hsv[2])

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
        hist, _ = np.histogram(channel, bins=16, range=(0, 256), density=True)
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

    if len(lesion_pixels_hsv) > n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(lesion_pixels_hsv)
        dominant_colors = kmeans.cluster_centers_.astype(int).tolist()
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
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lesion_h = hsv[..., 0][mask > 0]
    lesion_s = hsv[..., 1][mask > 0]
    lesion_v = hsv[..., 2][mask > 0]

    ys, xs = np.where(mask > 0)
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
        features["delta_H_center_periphery"] = float(np.mean(center_h) - np.mean(periph_h))
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
    mask_bin = (mask > 0).astype(np.uint8)
    area = np.sum(mask_bin)
    features["area"] = int(area)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = cv2.arcLength(contours[0], True) if contours else 0
    if contours:
        features["perimeter"] = float(perimeter)
    if area > 0:
        features["perimeter_area_ratio"] = float(perimeter / np.sqrt(area))
    if perimeter > 0:
        features["circularity"] = float(4 * np.pi * area / (perimeter**2))

    props_list = regionprops(mask_bin)
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
    mask_bin = (mask > 0).astype(np.uint8)
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
    if len(counts) >= 2:
        coeffs = np.polyfit(np.log(1 / sizes), np.log(counts), 1)
        features["fractal_dimension"] = float(-coeffs[0])
    return features


def extract_texture_features(image: np.ndarray, mask: np.ndarray) -> dict:
    """Извлечение текстурных признаков (GLCM + LBP)."""
    features = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lesion_gray = gray * (mask > 0)
    if np.sum(mask) == 0:
        return features

    lesion_norm = (lesion_gray / 16).astype(np.uint8)
    glcm = graycomatrix(
        lesion_norm,
        distances=[1, 2, 4],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=16,
        symmetric=True,
        normed=True,
    )
    contrast = graycoprops(glcm, "contrast").mean()
    homogeneity = graycoprops(glcm, "homogeneity").mean()
    energy = graycoprops(glcm, "energy").mean()
    p = glcm.flatten()
    features["glcm_contrast"] = float(contrast)
    features["glcm_homogeneity"] = float(homogeneity)
    features["glcm_energy"] = float(energy)
    features["glcm_entropy"] = float(entropy(p + 1e-6))

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lesion_lbp = lbp[mask > 0].astype(int)
    hist, _ = np.histogram(lesion_lbp, bins=np.arange(0, 11), density=True)
    features["lbp_uniformity"] = float(np.max(hist))
    features["lbp_entropy"] = float(entropy(hist + 1e-6))
    features["lbp_mean"] = float(np.mean(lesion_lbp))
    features["lbp_std"] = float(np.std(lesion_lbp))
    features["lbp_median"] = float(np.median(lesion_lbp))
    return features