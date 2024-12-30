# core.py
import os
import torch
import matplotlib.pyplot as plt
import datetime
from typing import Union

from PIL import Image
from glob import glob

from IPython.display import display
from IPython.display import Image as IPImage
from tqdm.autonotebook import tqdm

from sentence_transformers import SentenceTransformer, util


def display_similarity_results(target_image_path: str, similar_images: list[str], similarity_scores: list[float], save_path: str = None, threshold: float = 0.5, top_n: int = 10) -> str:
    """
    Display or save similarity results visualization.
    
    Args:
        target_image_path: Path to the target image
        similar_images: List of paths to similar images
        similarity_scores: List of similarity scores for each image
        save_path: If provided, save the plot to this path instead of displaying
        
    Returns:
        Path to saved plot if save_path is provided, None otherwise
    """
    n_similar = len(similar_images)
    n_cols = 4  # Number of columns in the grid
    n_rows = (n_similar + n_cols - 1) // n_cols + 1  # +1 for target image row

    # Create figure with proper size and spacing
    fig = plt.figure(figsize=(15, 4 * n_rows))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Display target image in the first row
    plt.subplot(n_rows, 1, 1)
    target_img = Image.open(target_image_path)
    plt.imshow(target_img)
    plt.title(f'Target Image\n{os.path.basename(target_image_path)}', size=12, pad=10)
    plt.axis('off')

    # Display similar images in a grid
    for idx, (img_path, score) in enumerate(zip(similar_images, similarity_scores)):
        plt.subplot(n_rows, n_cols, n_cols + idx + 1)
        img = Image.open(img_path)
        plt.imshow(img)
        title = f'Similarity: {score:.2f}\n{os.path.basename(img_path)}'
        plt.title(title, size=10)
        plt.xlabel(img_path, size=8, wrap=True)
        plt.axis('off')

    plt.suptitle(f'Image Similarity Results - Threshold: {threshold}, Top N: {top_n}', size=14, y=0.95)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        return save_path
    else:
        plt.show()
        plt.close(fig)
        return None


class Core:
    def __init__(self, model_name='clip-ViT-B-32', image_paths: list[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        self.embeddings = None
        self.image_paths = image_paths

        if self.image_paths is not None:
            self.embeddings = self.encode_images(self.image_paths)

    def encode_image(self, image_path: str) -> torch.Tensor:
        embedding = self.model.encode(Image.open(image_path), convert_to_tensor=True)
        return embedding.to(self.device)
    
    def encode_images(self, image_paths: list[str]) -> torch.Tensor:
        self.image_paths = image_paths
        embeddings = self.model.encode(
            [Image.open(filepath) for filepath in self.image_paths],
            batch_size=128,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        return embeddings.to(self.device)
    
    def community_detection(self, embeddings, threshold, min_community_size=10, init_max_size=100) -> list[list[int]]:
        """
        Function for Fast Community Detection

        Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).

        Returns only communities that are larger than min_community_size. The communities are returned
        in decreasing order. The first element in each list is the central point in the community.
        """

        # Compute cosine similarity scores
        cos_scores = util.cos_sim(embeddings, embeddings)

        # Minimum size for a community
        top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

        # Filter for rows >= min_threshold
        extracted_communities = []
        for i in range(len(top_k_values)):
            if top_k_values[i][-1] >= threshold:
                new_cluster = []

                # Only check top k most similar entries
                top_val_large, top_idx_large = cos_scores[i].topk(k=init_max_size, largest=True)
                top_idx_large = top_idx_large.tolist()
                top_val_large = top_val_large.tolist()

                if top_val_large[-1] < threshold:
                    for idx, val in zip(top_idx_large, top_val_large):
                        if val < threshold:
                            break

                        new_cluster.append(idx)
                else:
                    # Iterate over all entries (slow)
                    for idx, val in enumerate(cos_scores[i].tolist()):
                        if val >= threshold:
                            new_cluster.append(idx)

                extracted_communities.append(new_cluster)

        # Largest cluster first
        extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

        # Step 2) Remove overlapping communities
        unique_communities = []
        extracted_ids = set()

        for community in extracted_communities:
            add_cluster = True
            for idx in community:
                if idx in extracted_ids:
                    add_cluster = False
                    break

            if add_cluster:
                unique_communities.append(community)
                for idx in community:
                    extracted_ids.add(idx)

        return unique_communities
    
    def run_community_detection(self, embeddings: torch.Tensor, threshold: float, min_community_size: int = 10, init_max_size: int = 100) -> list[list[int]]:
        """
        Run community detection on a list of embeddings.
        """
        return self.community_detection(embeddings, threshold, min_community_size, init_max_size)

    def run_community_detection_from_images(self, image_paths: list[str], threshold: float, min_community_size: int = 10, init_max_size: int = 100) -> list[list[int]]:
        """
        Run community detection on a list of images.
        """
        if self.embeddings is None:
            print("Encoding images...")
            self.embeddings = self.encode_images(image_paths)
        return self.run_community_detection(self.embeddings, threshold, min_community_size, init_max_size)
    
    def run_cosine_similarity(self, target_embedding: torch.Tensor) -> torch.Tensor:
        """
        Run cosine similarity on a list of images.
        """
        return util.cos_sim(target_embedding, self.embeddings)

    def find_pairs(self, threshold: float, target_image_path: str, top_n: int = 10, return_json: bool = False) -> Union[tuple[list[str], list[float]], dict]:
        """
        Find pairs of images that are similar to the target image.
        
        Args:
            threshold: Minimum similarity score to consider
            target_image_path: Path to the target image
            top_n: Number of similar images to return
            return_json: If True, returns a JSON-compatible dictionary instead of tuple
            
        Returns:
            If return_json is False:
                Tuple containing:
                - List of paths to similar images
                - List of similarity scores
            If return_json is True:
                Dictionary containing all similarity results
        """
        self.last_threshold = threshold  # Store threshold for metadata
        target_embedding = self.encode_image(target_image_path)
        print("Running cosine similarity...")
        
        cosine_scores = self.run_cosine_similarity(target_embedding)
        
        print("Finding pairs...")
        pairs = []
        for i in range(len(cosine_scores[0])):
            if cosine_scores[0][i] >= threshold:
                pairs.append({'index': i, 'score': cosine_scores[0][i].item()})

        pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

        similar_images = []
        similarity_scores = []
        for pair in pairs[0:top_n]:
            similar_images.append(self.image_paths[pair['index']])
            similarity_scores.append(pair['score'])
        
        if return_json:
            return self.create_similarity_results_json(target_image_path, similar_images, similarity_scores)
        
        return similar_images, similarity_scores
    
    def create_similarity_results_json(self, target_image_path: str, similar_images: list[str], similarity_scores: list[float]) -> dict:
        """
        Create a JSON-compatible dictionary with similarity results.
        
        Args:
            target_image_path: Path to the target image
            similar_images: List of paths to similar images
            similarity_scores: List of similarity scores for each image
            
        Returns:
            Dictionary containing all similarity results and metadata
        """
        results = {
            "target_image": {
                "path": target_image_path,
                "filename": os.path.basename(target_image_path)
            },
            "total_results": len(similar_images),
            "similar_images": [
                {
                    "path": img_path,
                    "filename": os.path.basename(img_path),
                    "similarity_score": float(score),  # Convert to float for JSON serialization
                    "rank": idx + 1
                }
                for idx, (img_path, score) in enumerate(zip(similar_images, similarity_scores))
            ],
            "metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "threshold": self.last_threshold if hasattr(self, 'last_threshold') else None
            }
        }
        
        return results
    
    
if __name__ == "__main__":
    
    image_test = "../data/test_stg1/img_03670.jpg"
    images_list = glob("../data/test_stg1/*.jpg")
    
    core = Core(image_paths=images_list)
    
    # Get similar images and their scores
    similar_images, similarity_scores = core.find_pairs(0.5, image_test, 10)
    
    # Display results using matplotlib
    display_similarity_results(image_test, similar_images, similarity_scores)
