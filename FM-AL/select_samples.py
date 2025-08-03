import argparse
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.feature_sampling_utils import *
from sklearn.cluster import KMeans, Birch
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.special import softmax

GAMMA_VALUES = [0.1, 1]
DEFAULT_RANDOM_STATE = 12345
USL_CONFIG = {"num_iters": 10, "k": 5, "alpha": 2, "epsilon": 1e-6, "lambda_reg": 0.5}
USL_T_CONFIG = {
    "k": 5,
    "alpha": 2,
    "num_iters": 3,
    "lambda_reg": 0.5,
    "t": 0.25,
    "tau": 0.5,
    "epsilon": 1e-6,
}
PROBCOVER_ALPHA = 0.95


def load_npy_features(folder_path):
    """
    Load all .npy files from a folder into a feature array and extract case_i_slice_j names.

    Args:
        folder_path (str): Path to the folder containing .npy files.

    Returns:
        tuple: (feats: ndarray of shape [n, p], name_list: ndarray of str of shape [n])
    """
    npy_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".npy"))
    feats, name_list = [], []

    for fname in npy_files:
        file_path = os.path.join(folder_path, fname)
        vec = np.load(file_path)

        if vec.ndim != 1:
            print(f"⚠️ Skipping file (non-1D vector): {fname}")
            continue

        feats.append(vec)
        match = re.search(r"(case_\d+_slice_\d+)", fname)
        name_list.append(match.group(1) if match else os.path.splitext(fname)[0])

    return np.stack(feats), np.array(name_list)


def load_npy_features_from_csv(csv_path, fm_name):
    """
    Load .npy features from paths in a CSV file, adapting paths to feature file locations.

    Args:
        csv_path (str): Path to CSV file with 'image_pth' column.
        fm_name (str): Feature extractor name (e.g., 'clip_RN50x64').
        organ (str, optional): Organ name for path construction.

    Returns:
        tuple: (feats: ndarray of shape [n, p], name_list: ndarray of str of shape [n])
    """
    df = pd.read_csv(csv_path)
    if "image_pth" not in df.columns:
        raise ValueError("CSV file missing 'image_pth' column")

    feats, name_list = [], []
    base_path = "../data/feature"

    for raw_path in df["image_pth"]:
        match = re.search(r"/dataset/([^/]+)/([^/]+)\.npy$", raw_path)
        if not match:
            print(f"⚠️ Unable to parse path, skipping: {raw_path}")
            continue

        organ_name, case_slice = match.group(1), match.group(2)
        new_filename = f"{case_slice}_{fm_name}.npy"
        feature_path = os.path.join(base_path, organ_name, fm_name, new_filename)

        if not os.path.isfile(feature_path):
            print(f"⚠️ Feature file not found, skipping: {feature_path}")
            continue

        vec = np.load(feature_path)
        print(vec.shape)
        if vec.ndim != 1:
            print(f"⚠️ Skipping file (non-1D vector): {feature_path}")
            continue

        feats.append(vec)
        name_list.append(case_slice)

    return np.stack(feats), np.array(name_list)


# Kernel-based clustering functions
def construct_kernels_without_uncertainty(X):
    """
    Construct Gaussian kernels for input features using multiple gamma values.

    Args:
        X (ndarray): Input features of shape [n, p].

    Returns:
        list: List of kernel matrices.
    """
    kernels = []
    for gamma in GAMMA_VALUES:
        pairwise_dist = pairwise_distances(X, metric="euclidean")
        kernel = np.exp(-gamma * pairwise_dist**2 / 2)
        kernels.append(kernel)
    return kernels


def multiple_kernel_kmeans(kernels, k, max_iter=10):
    """
    Perform k-means clustering using multiple kernel matrices.

    Args:
        kernels (list): List of kernel matrices.
        k (int): Number of clusters.
        max_iter (int): Maximum number of iterations.

    Returns:
        tuple: (centers: ndarray, cluster_assignments: ndarray)
    """
    n_samples = kernels[0].shape[0]
    cluster_assignments = np.random.choice(k, n_samples)

    for iteration in range(max_iter):
        combined_kernel = sum(kernels)
        new_assignments = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            distances = np.zeros(k)
            for j in range(k):
                in_cluster = cluster_assignments == j
                cluster_size = np.sum(in_cluster)
                if cluster_size == 0:
                    distances[j] = np.inf
                    continue
                cluster_indices = np.where(in_cluster)[0]
                kernel_values = combined_kernel[i, cluster_indices]
                center_dist = (
                    combined_kernel[i, i]
                    - 2 * np.sum(kernel_values) / cluster_size
                    + np.sum(combined_kernel[np.ix_(cluster_indices, cluster_indices)])
                    / (cluster_size**2)
                )
                distances[j] = center_dist
            new_assignments[i] = np.argmin(distances)

        for j in range(k):
            if np.sum(new_assignments == j) == 0:
                new_assignments[np.random.choice(n_samples)] = j

        if np.array_equal(cluster_assignments, new_assignments):
            break
        cluster_assignments = new_assignments

    centers = np.zeros((k, n_samples))
    for j in range(k):
        in_cluster = cluster_assignments == j
        if np.sum(in_cluster) == 0:
            continue
        cluster_indices = np.where(in_cluster)[0]
        cluster_points = combined_kernel[cluster_indices][:, cluster_indices]
        centers[j, cluster_indices] = np.mean(cluster_points, axis=0)

    return centers, cluster_assignments


def calculate_typicality(cluster_points):
    """
    Calculate typicality scores for points in a cluster based on cosine distances to centroid.

    Args:
        cluster_points (ndarray): Points in a cluster.

    Returns:
        ndarray: Typicality scores.
    """
    centroid = np.mean(cluster_points, axis=0)
    distances = cosine_distances(cluster_points, [centroid])[:, 0]
    return 1 / (distances + 1e-10)


# Sample selection strategies
def select_valuable_samples(AL_method, feats, name_list, num_samples):
    """
    Select valuable samples based on the specified active learning plan.

    Args:
        AL_method (str): Name of the active learning plan.
        feats (ndarray): Feature array of shape [n, p].
        name_list (ndarray): Array of sample names.
        num_samples (int): Number of samples to select.

    Returns:
        list: Selected sample names.
    """
    if AL_method == "ALPS":
        kmeans = KMeans(n_clusters=num_samples, random_state=DEFAULT_RANDOM_STATE)
        kmeans.fit(feats)
        centers, labels = kmeans.cluster_centers_, kmeans.labels_
        return [
            name_list[
                cluster_indices[
                    np.argmin(
                        np.linalg.norm(feats[cluster_indices] - centers[i], axis=1)
                    )
                ]
            ]
            for i in range(num_samples)
            if (cluster_indices := np.where(labels == i)[0]).size > 0
        ]

    elif AL_method == "FPS":
        num_clusters = num_samples // 2
        extra_sample = num_samples % 2 == 1
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(feats)
        labels = kmeans.labels_
        select_samples = []

        for i in range(num_clusters):
            cluster_indices = np.where(labels == i)[0]
            if cluster_indices.size == 0:
                continue
            cluster_feats = feats[cluster_indices]
            distances = cdist(cluster_feats, cluster_feats, metric="euclidean")
            max_dist_indices = np.unravel_index(np.argmax(distances), distances.shape)
            select_samples.extend(
                [
                    name_list[cluster_indices[max_dist_indices[0]]],
                    name_list[cluster_indices[max_dist_indices[1]]],
                ]
            )

        if extra_sample and num_clusters > 0:
            select_samples.append(name_list[np.where(labels == num_clusters - 1)[0][0]])
        return select_samples[:num_samples]

    elif AL_method == "USL":
        n = feats.shape[0]
        nn = NearestNeighbors(n_neighbors=USL_CONFIG["k"] + 1, algorithm="auto").fit(
            feats
        )
        distances, indices = nn.kneighbors(feats)
        density = 1 / (np.mean(distances[:, 1:], axis=1) + USL_CONFIG["epsilon"])
        selected_samples, reg_values = [], np.zeros(n)

        for _ in range(USL_CONFIG["num_iters"]):
            if selected_samples:
                selected_feats = feats[selected_samples]
                dists = np.linalg.norm(
                    feats[:, None, :] - selected_feats[None, :, :], axis=2
                )
                safe_dists = np.clip(dists, a_min=1e-2, a_max=None)
                reg_values = np.sum(
                    1 / ((safe_dists ** USL_CONFIG["alpha"]) + USL_CONFIG["epsilon"]),
                    axis=1,
                )
            utility = density - USL_CONFIG["lambda_reg"] * reg_values
            selected_samples = np.argsort(utility)[-num_samples:].tolist()

        return [name_list[i] for i in selected_samples]

    elif AL_method == "USL-T":
        n = feats.shape[0]
        centroids = feats[np.random.choice(n, num_samples, replace=False)]
        nn = NearestNeighbors(n_neighbors=USL_T_CONFIG["k"] + 1, algorithm="auto").fit(
            feats
        )
        distances, indices = nn.kneighbors(feats)
        nearest_neighbors = indices[:, 1:]

        soft_assignments = np.zeros((n, num_samples))
        for _ in range(USL_T_CONFIG["num_iters"]):
            similarities = np.dot(feats, centroids.T)
            soft_assignments = softmax(similarities / USL_T_CONFIG["t"], axis=1)
            confident_indices = np.max(soft_assignments, axis=1) >= USL_T_CONFIG["tau"]
            confident_assignments = soft_assignments[confident_indices]
            confident_feats = feats[confident_indices]

            for j in range(num_samples):
                cluster_weights = confident_assignments[:, j]
                weighted_sum = np.sum(
                    cluster_weights[:, None] * confident_feats, axis=0
                )
                centroids[j] = weighted_sum / (
                    np.sum(cluster_weights) + USL_T_CONFIG["epsilon"]
                )

            local_regularization = np.zeros((n, num_samples))
            for i in range(n):
                neighbor_avg = np.mean(soft_assignments[nearest_neighbors[i]], axis=0)
                local_regularization[i] = softmax(neighbor_avg / USL_T_CONFIG["t"])
            combined_assignments = (
                USL_T_CONFIG["lambda_reg"] * local_regularization + soft_assignments
            )
            cluster_confidences = np.max(combined_assignments, axis=1)

        return [name_list[i] for i in np.argsort(-cluster_confidences)[:num_samples]]

    elif AL_method == "Typiclust":
        kmeans = KMeans(n_clusters=num_samples, random_state=0)
        kmeans.fit(feats)
        labels = kmeans.labels_
        return [
            name_list[
                cluster_indices[np.argmax(calculate_typicality(feats[cluster_indices]))]
            ]
            for i in range(num_samples)
            if (cluster_indices := np.where(labels == i)[0]).size > 0
        ]

    elif AL_method == "Probcover":
        delta = estimate_delta(feats, num_samples, PROBCOVER_ALPHA)
        nn = NearestNeighbors(radius=delta, algorithm="auto").fit(feats)
        distances, indices = nn.radius_neighbors(feats)
        adjacency_list = {i: set(neigh) for i, neigh in enumerate(indices)}
        selected_indices, covered_set = [], set()

        for _ in range(num_samples):
            if not adjacency_list:
                break
            out_degrees = {
                i: len(neigh - covered_set) for i, neigh in adjacency_list.items()
            }
            max_out_degree_index = max(out_degrees, key=out_degrees.get)
            selected_indices.append(max_out_degree_index)
            covered_set.update(adjacency_list[max_out_degree_index])

        return [name_list[i] for i in selected_indices]

    elif AL_method == "CALR":
        birch = Birch(n_clusters=num_samples)
        birch.fit(feats)
        labels = birch.labels_
        return [
            name_list[
                cluster_indices[
                    np.argmax(compute_information_density(feats[cluster_indices]))
                ]
            ]
            for i in range(num_samples)
            if (cluster_indices := np.where(labels == i)[0]).size > 0
        ]

    elif AL_method == "Coreset":
        n_samples = len(name_list)
        dist_mat = np.sqrt(
            np.clip(
                -2 * np.matmul(feats, feats.T)
                + np.diag(np.matmul(feats, feats.T)).reshape(-1, 1)
                + np.diag(np.matmul(feats, feats.T)).reshape(1, -1),
                0,
                None,
            )
        )
        labeled_idxs = np.zeros(n_samples, dtype=bool)
        first_sample_idx = np.random.choice(n_samples)
        labeled_idxs[first_sample_idx] = True
        min_distances = dist_mat[:, first_sample_idx]
        select_samples = [name_list[first_sample_idx]]

        for _ in tqdm(range(num_samples - 1), desc="Coreset selection", ncols=100):
            next_sample_idx = np.argmax(min_distances)
            labeled_idxs[next_sample_idx] = True
            select_samples.append(name_list[next_sample_idx])
            min_distances = np.minimum(min_distances, dist_mat[:, next_sample_idx])

        return select_samples

    elif AL_method == "URDS_Unc":
        kernels = construct_kernels_without_uncertainty(feats)
        centers, labels = multiple_kernel_kmeans(kernels, num_samples)
        selected_indices = [
            np.where(
                name_list
                == name_list[labels == i][
                    np.argmax(calculate_typicality(feats[labels == i]))
                ]
            )[0][0]
            for i in range(num_samples)
            if (feats[labels == i]).size > 0
        ]
        return [name_list[i] for i in selected_indices]

    elif AL_method == "BAL":
        kmeans = KMeans(n_clusters=100, random_state=0).fit(feats)
        distances = cdist(feats, kmeans.cluster_centers_, metric="euclidean")
        complexity_scores = (
            np.sort(distances, axis=1)[:, 1] - np.sort(distances, axis=1)[:, 0]
        )
        return [name_list[i] for i in np.argsort(-complexity_scores)[:num_samples]]

    elif AL_method == "RepDiv":

        def calculate_similarity_matrix(X):
            X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
            return np.dot(X_norm, X_norm.T)

        def select_representative_indices(feats, num_select, sim_matrix):
            n = feats.shape[0]
            selected, unselected = [], list(range(n))
            max_sim = np.zeros(n)
            for _ in range(num_select):
                best_gain, best_idx = -np.inf, -1
                for idx in unselected:
                    temp_sim = np.maximum(max_sim, sim_matrix[idx])
                    if (gain := np.sum(temp_sim)) > best_gain:
                        best_gain, best_idx = gain, idx
                selected.append(best_idx)
                unselected.remove(best_idx)
                max_sim = np.maximum(max_sim, sim_matrix[best_idx])
            return selected

        def select_diverse_indices(feats, selected, num_needed, sim_matrix):
            dis_matrix = 1 - sim_matrix
            candidates = set(range(feats.shape[0])) - set(selected)
            for _ in range(num_needed):
                best_gain, best_idx = -np.inf, -1
                for idx in candidates:
                    if (
                        diversity_score := np.max(dis_matrix[idx, selected])
                    ) > best_gain:
                        best_gain, best_idx = diversity_score, idx
                selected.append(best_idx)
                candidates.remove(best_idx)
            return selected

        sim_matrix = calculate_similarity_matrix(feats)
        num_rep = int(num_samples * 0.7)
        rep_indices = select_representative_indices(feats, num_rep, sim_matrix)
        final_indices = select_diverse_indices(
            feats, rep_indices, num_samples - num_rep, sim_matrix
        )
        return [name_list[i] for i in final_indices]

    elif AL_method.startswith("Random_seed_"):
        seed = int(AL_method.split("_")[-1])
        np.random.seed(seed)
        return np.random.choice(name_list, num_samples, replace=False).tolist()

    else:
        raise ValueError(f"Invalid plan name: {AL_method}")


# Main active learning plan generation
def al_generation_plan(AL_method, organ, feats_file_path, fm_name, num_samples):
    """
    Generate an active learning plan and save selected sample paths to a CSV file.

    Args:
        AL_method (str): Active learning plan name.
        organ (str): Organ or dataset name.
        feats_file_path (str): Path to CSV file with feature paths.
        fm_name (str): Feature extractor name.
        num_samples (int): Number of samples to select.
    """
    np.random.seed(DEFAULT_RANDOM_STATE)

    feats, name_list = load_npy_features_from_csv(feats_file_path, fm_name)

    select_samples_list = select_valuable_samples(
        AL_method, feats, name_list, num_samples
    )

    image_paths = [
        os.path.join(f"../data/dataset/{organ}/{pid}.npy")
        for pid in select_samples_list
    ]
    mask_paths = [
        os.path.join(f"../data/dataset/{organ}/{pid}_label.npy")
        for pid in select_samples_list
    ]

    print(f"Selected:\n{image_paths}")
    os.makedirs(f"../data/AL_Plan/{organ}/{fm_name}", exist_ok=True)
    save_path = f"../data/AL_Plan/{organ}/{fm_name}/{AL_method}_{num_samples}.csv"
    pd.DataFrame({"image_pth": image_paths, "mask_pth": mask_paths}).to_csv(
        save_path, index=False
    )
    print(f"Paths saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Active Learning Plan Generation")
    parser.add_argument(
        "--foundation_models",
        nargs="+",
        default=[
            "resnet_resnet18",
            "clip_RN50x64",
            "medclip_ViT",
            "sam_vit_b",
            "medsam_medsam_vit_b",
            "sam2_2b+",
            "sam2_2.1b+",
            "dino_ViT-B_16",
            "dino2_ViT-B_14",
        ],
        help="List of foundation model names",
    )
    parser.add_argument(
        "--organs",
        nargs="+",
        default=[
            "Heart_preprocessed",
            "Spleen_preprocessed",
            "Kvasir_preprocessed",
            "TN3K_preprocessed",
            "Derma",
            "Breast",
            "Pneumonia",
        ],
        help="List of organ or dataset names",
    )
    parser.add_argument(
        "--annotation_budgets",
        nargs="+",
        type=int,
        action="append",
        default=[
            [38, 76, 114],
            [51, 101, 254],
            [90, 135, 180],
            [70, 140, 210],
            [129, 194, 258],
            [27, 54, 81],
            [47, 94, 141],
        ],
        help="List of annotation budgets for each organ",
    )
    parser.add_argument(
        "--AL_methods",
        nargs="+",
        default=["FPS", "Typiclust", "Probcover", "Coreset", "ALPS", "RepDiv", "BAL"],
        help="List of active learning methods",
    )
    args = parser.parse_args()


    for organ, budgets in zip(args.organs, args.annotation_budgets):
        feats_file_path = f"../data/dataset/{organ}/splits/train.csv"
        for fm in args.foundation_models:
            for AL_method in args.AL_methods:
                for budget in budgets:
                    al_generation_plan(AL_method, organ, feats_file_path, fm, budget)
                    print(
                        f"{AL_method} AL Generation Plan Done for model {fm} with budget {budget}"
                    )
