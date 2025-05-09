import torch.nn as nn
import numpy as np
import torch
from tqdm import tqdm
import clip
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import pairwise_distances, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap.umap_ as umap
from collections import Counter
from PIL import Image
from scipy.stats import entropy
import json
from PIL import Image
from collections import defaultdict
import os



class TripletNetwork(nn.Module):
    def __init__(self, clip_model, device, embedding_dim=512):
        super(TripletNetwork, self).__init__()
        self.clip_model = clip_model.to(device)
        self.fc = nn.Linear(clip_model.visual.output_dim, embedding_dim)

    def forward_once(self, x):
        x = self.clip_model.encode_image(x)
        return self.fc(x)

    def forward(self, anchor, positive, negative):
        output1 = self.forward_once(anchor)
        output2 = self.forward_once(positive)
        output3 = self.forward_once(negative)

        return output1, output2, output3


class Inference:
    def __init__(self, model_path, clip_model_name="ViT-B/32", device="cpu"):
        self.device = device
        self.clip_model, self.preprocess = clip.load(clip_model_name, device=device)
        self.clip_model.eval()
        self.model = self.load_model(model_path).to(self.device)

    def load_model(self, model_path):
        model = TripletNetwork(self.clip_model, self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def get_embedding(self, image_path, resize=True, target_size = (770, 1048)):
        image = Image.open(image_path).convert("RGB")
        if resize:
            image = image.resize(target_size)
        processed_image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_embedding = self.model.forward_once(processed_image)
        return image_embedding

    def get_all_embeddings(self, image_list, resize=True, target_size = (770, 1048)):
        embeddings = []
        for img in tqdm(image_list):
            emb = self.get_embedding(img)
            embeddings.append(emb.cpu().numpy().squeeze())
        return np.stack(embeddings)

    @staticmethod
    def compute_intradiversity(embeddings):
        distances = pairwise_distances(embeddings)
        triu = np.triu(distances, k=1)
        mean_diversity = triu[triu != 0].mean()
        return mean_diversity

    @staticmethod
    def mean_distance_between_datasets(emb1, emb2):
        mean1 = emb1.mean(axis=0)
        mean2 = emb2.mean(axis=0)
        return np.linalg.norm(mean1 - mean2)

    @staticmethod
    def plot_tsne(emb1, emb2):
        all_embeddings = np.vstack([emb1, emb2])
        labels = ['synthetic'] * len(emb1) + ['real'] * len(emb2)

        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        reduced = tsne.fit_transform(all_embeddings)

        plt.figure(figsize=(8, 6))
        for label in set(labels):
            idxs = [i for i, l in enumerate(labels) if l == label]
            plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=label, alpha=0.6)
        plt.legend()
        plt.title("t-SNE Projection of Document Embeddings")
        plt.show()


def compute_intradiversity(embeddings):
    distances = pairwise_distances(embeddings)
    triu = np.triu(distances, k=1)
    mean_diversity = triu[triu != 0].mean()
    return mean_diversity


def mean_distance_between_datasets(emb1, emb2):
    mean1 = emb1.mean(axis=0)
    mean2 = emb2.mean(axis=0)
    return np.linalg.norm(mean1 - mean2)


def plot_tsne(emb1, emb2):
    all_embeddings = np.vstack([emb1, emb2])
    labels = ['synthetic'] * len(emb1) + ['real'] * len(emb2)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(all_embeddings)

    plt.figure(figsize=(6, 4))
    for label in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=label, alpha=0.6)
    plt.legend()
    plt.title("t-SNE Projection of Document Embeddings")
    plt.show()


def compute_pairwise_distances(embeddings):
    dists = pairwise_distances(embeddings)
    upper_tri_indices = np.triu_indices_from(dists, k=1)
    pair_dists = dists[upper_tri_indices]
    return pair_dists, dists


def plot_histogram(pair_dists, title="Histogram of Pairwise Distances"):
    plt.figure()
    plt.hist(pair_dists, bins=50, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.show()


def plot_boxplot(pair_dists, title="Boxplot of Pairwise Distances"):
    plt.figure()
    plt.boxplot(pair_dists)
    plt.title(title)
    plt.ylabel("Distance")
    plt.show()


def plot_cdf(pair_dists, title="CDF of Pairwise Distances"):
    sorted_dists = np.sort(pair_dists)
    cdf = np.arange(len(sorted_dists)) / float(len(sorted_dists))
    plt.figure()
    plt.plot(sorted_dists, cdf, lw=2)
    plt.title(title)
    plt.xlabel("Distance")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.show()


def compute_clustering_metrics(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    sil_score = silhouette_score(embeddings, labels)
    ch_score = calinski_harabasz_score(embeddings, labels)
    db_score = davies_bouldin_score(embeddings, labels)
    return labels, sil_score, ch_score, db_score

def plot_umap(embeddings, labels=None, title="UMAP Projection"):
    reducer = umap.UMAP(random_state=42)
    embedding_umap = reducer.fit_transform(embeddings)
    plt.figure()
    if labels is None:
        plt.scatter(embedding_umap[:, 0], embedding_umap[:, 1], alpha=0.7)
    else:
        plt.scatter(embedding_umap[:, 0], embedding_umap[:, 1], c=labels, cmap="viridis", alpha=0.7)
    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.show()

def plot_pca(embeddings, labels=None, title="PCA Projection"):
    pca = PCA(n_components=2)
    embedding_pca = pca.fit_transform(embeddings)
    plt.figure()
    if labels is None:
        plt.scatter(embedding_pca[:, 0], embedding_pca[:, 1], alpha=0.7)
    else:
        plt.scatter(embedding_pca[:, 0], embedding_pca[:, 1], c=labels, cmap="viridis", alpha=0.7)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


def plot_heatmap(distance_matrix, title="Heatmap of Distance Matrix"):
    plt.figure(figsize=(6, 4))
    sns.heatmap(distance_matrix, cmap="viridis")
    plt.title(title)
    plt.xlabel("Document index")
    plt.ylabel("Document index")
    plt.show()

def plot_jointplot(embeddings, title="Joint Plot of PCA"):
    pca = PCA(n_components=2)
    embedding_pca = pca.fit_transform(embeddings)
    df = pd.DataFrame({
        'PC1': embedding_pca[:, 0],
        'PC2': embedding_pca[:, 1]
    })
    joint_plot = sns.jointplot(data=df, x='PC1', y='PC2', kind='scatter', height=6, alpha=0.7)
    joint_plot.fig.suptitle(title, y=1.02)
    plt.show()


def compare_datasets(embeddings_synth, embeddings_real, name_1, name_2):
    synth_pair_dists, synth_distance_matrix = compute_pairwise_distances(embeddings_synth)
    real_pair_dists, real_distance_matrix = compute_pairwise_distances(embeddings_real)

    plot_histogram(synth_pair_dists, title=f"{name_1}: Histogram of Pairwise Distances")
    plot_histogram(real_pair_dists, title=f"{name_2}: Histogram of Pairwise Distances")

    plot_boxplot(synth_pair_dists, title=f"{name_1}: Boxplot of Pairwise Distances")
    plot_boxplot(real_pair_dists, title=f"{name_2}: Boxplot of Pairwise Distances")

    plot_cdf(synth_pair_dists, title=f"{name_1}: CDF of Pairwise Distances")
    plot_cdf(real_pair_dists, title=f"{name_2}: CDF of Pairwise Distances")

    synth_labels, synth_sil, synth_ch, synth_db = compute_clustering_metrics(embeddings_synth, n_clusters=5)
    real_labels, real_sil, real_ch, real_db = compute_clustering_metrics(embeddings_real, n_clusters=5)

    print(f"{name_1} dataset clustering metrics:")
    print("Silhouette Score: ", synth_sil)
    print("Calinski-Harabasz Score: ", synth_ch)
    print("Davies-Bouldin Score: ", synth_db)
    print(f"\n{name_2} dataset clustering metrics:")
    print("Silhouette Score: ", real_sil)
    print("Calinski-Harabasz Score: ", real_ch)
    print("Davies-Bouldin Score: ", real_db)

    plot_umap(embeddings_synth, title=f"{name_1}: UMAP Projection")
    plot_umap(embeddings_real, title=f"{name_2}: UMAP Projection")

    plot_pca(embeddings_synth, title=f"{name_1}: PCA Projection")
    plot_pca(embeddings_real, title=f"{name_2}: PCA Projection")

    plot_heatmap(synth_distance_matrix, title=f"{name_1}: Heatmap of Distance Matrix")
    plot_heatmap(real_distance_matrix, title=f"{name_2}: Heatmap of Distance Matrix")

    plot_jointplot(embeddings_synth, title=f"{name_1}: Joint Plot of PCA")
    plot_jointplot(embeddings_real, title=f"{name_2}: Joint Plot of PCA")


def aggregate_bbox_centers_scaled(bboxes):
    centers = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        center_x = (x_min + x_max) / 2.0 / 770
        center_y = 1 - (y_min + y_max) / 2.0 / 1048
        centers.append((center_x, center_y))
    return np.array(centers)


def plot_bbox_center_heatmap(bboxes, bins=50):
    centers = aggregate_bbox_centers_scaled(bboxes)
    heatmap, xedges, yedges = np.histogram2d(centers[:, 0], centers[:, 1],
                                             bins=bins, range=[[0, 1], [0, 1]])
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap.T, origin='lower', extent=[0, 1, 0, 1], cmap='hot')
    plt.title("Heatmap центров bbox-ов (нормализованные координаты)")
    plt.xlabel("Нормализованная X")
    plt.ylabel("Нормализованная Y")
    plt.colorbar(label="Количество")
    plt.show()


def plot_bbox_size_distribution_scaled(bboxes):
    widths = []
    heights = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        bbox_width = (x_max - x_min) / 770
        bbox_height = (y_max - y_min) / 1048
        widths.append(bbox_width)
        heights.append(bbox_height)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=50, color='blue', alpha=0.7)
    plt.title("Распределение нормализованных ширин bbox-ов")
    plt.xlabel("Ширина")
    plt.ylabel("Частота")

    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=50, color='green', alpha=0.7)
    plt.title("Распределение нормализованных высот bbox-ов")
    plt.xlabel("Высота")
    plt.ylabel("Частота")

    plt.show()


def plot_heatmap_all(bboxes, title):
    centers = aggregate_bbox_centers_scaled(bboxes)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8])

    sns.kdeplot(
        x=centers[:, 0],
        y=centers[:, 1],
        cmap="Reds",
        fill=True,
        ax=ax
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    ax.set_title(title)
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    plt.savefig(f"heatmap_{title}.png", dpi=300, bbox_inches='tight')
    plt.show()


def compute_heatmap(bboxes, bins=10):
    centers = aggregate_bbox_centers_scaled(bboxes)
    heatmap, xedges, yedges = np.histogram2d(centers[:, 0], centers[:, 1],
                                             bins=bins, range=[[0, 1], [0, 1]])
    return heatmap, xedges, yedges


def compute_heatmap_entropy(bboxes):
    centers = aggregate_bbox_centers_scaled(bboxes)
    heatmap, _, _ = compute_heatmap(bboxes)
    h = heatmap.flatten()
    h = h / h.sum() if h.sum() > 0 else h
    return entropy(h)


def compute_average_pairwise_distance(bboxes):
    centers = aggregate_bbox_centers_scaled(bboxes)
    dists = pairwise_distances(centers)
    upper_tri = dists[np.triu_indices_from(dists, k=1)]
    return np.mean(upper_tri)


def calculate_coverage_ratio(bboxes, img_width=770, img_height=1048):
    total_area = img_width * img_height
    bbox_area = 0

    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        width = max(0, x_max - x_min)
        height = max(0, y_max - y_min)
        bbox_area += width * height

    coverage_ratio = (bbox_area / total_area) * 100
    return coverage_ratio


def calculate_dataset_coverage(documents, img_width=770, img_height=1048):
    ratios = [calculate_coverage_ratio(bbox, img_width, img_height) for bbox in documents]
    return np.mean(ratios)


def bbox_diversity_score(bboxes, grid_size=(4, 4), img_width=770, img_height=1048):
    grid_w, grid_h = grid_size
    grid = np.zeros(grid_size, dtype=int)

    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0

        grid_x = min(int(center_x / img_width * grid_w), grid_w - 1)
        grid_y = min(int(center_y / img_height * grid_h), grid_h - 1)

        grid[grid_y, grid_x] = 1

    filled_cells = np.sum(grid)
    total_cells = grid_w * grid_h

    diversity_score = (filled_cells / total_cells) * 100

    return diversity_score


def dataset_diversity_score(documents, grid_size=(4, 4), img_width=770, img_height=1048):
    scores = [bbox_diversity_score(bboxes, grid_size, img_width, img_height) for bboxes in documents]
    return np.mean(scores)


def document_grid_mask(bboxes, grid_size=(4, 4), img_width=770, img_height=1048):
    grid_w, grid_h = grid_size
    grid = np.zeros(grid_size, dtype=int)

    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0

        grid_x = min(int(center_x / img_width * grid_w), grid_w - 1)
        grid_y = min(int(center_y / img_height * grid_h), grid_h - 1)

        grid[grid_y, grid_x] = 1

    return grid.flatten()


def calculate_layout_uniqueness(documents, grid_size=(4, 4), img_width=770, img_height=1048):
    masks = [tuple(document_grid_mask(bboxes, grid_size, img_width, img_height)) for bboxes in documents]

    mask_counts = Counter(masks)

    unique_count = sum(1 for mask in masks if mask_counts[mask] == 1)

    uniqueness_percentage = (unique_count / len(documents)) * 100

    return uniqueness_percentage


def scale_bboxes(bbox, orig_width, orig_height, new_width=770, new_height=1048):
    scale_x = new_width / orig_width
    scale_y = new_height / orig_height

    x_min, y_min, x_max, y_max = bbox
    x_min_scaled = x_min * scale_x
    x_max_scaled = x_max * scale_x
    y_min_scaled = y_min * scale_y
    y_max_scaled = y_max * scale_y

    return [int(x_min_scaled), int(y_min_scaled), int(x_max_scaled), int(y_max_scaled)]


def get_bboxes(labels, labels_path, images_path):
    bboxes = []
    doc_bboxes = []
    pii_bboxes = []
    doc_pii_bboxes = []
    pii_bboxes_dict = defaultdict(list)
    for label in labels:
        img = Image.open(os.path.join(images_path, label.replace("json", "png")))
        orig_width, orig_height = img.size
        doc_bbox = []
        doc_pii_bbox = []
        with open(os.path.join(labels_path, label), "r") as f:
            data = json.load(f)
            for tag, bbox in zip(data["ner_tags"], data["bboxes"]):
                if tag != "O":
                    pii_bboxes.append(scale_bboxes(bbox, orig_width, orig_height))
                    pii_bboxes_dict[tag.split("-")[1]].append(scale_bboxes(bbox, orig_width, orig_height))
                    doc_pii_bbox.append(scale_bboxes(bbox, orig_width, orig_height))
                bboxes.append(scale_bboxes(bbox, orig_width, orig_height))
                doc_bbox.append(scale_bboxes(bbox, orig_width, orig_height))
        doc_bboxes.append(doc_bbox)
        doc_pii_bboxes.append(doc_pii_bbox)
    return bboxes, pii_bboxes, dict(pii_bboxes_dict), doc_bboxes, doc_pii_bboxes
