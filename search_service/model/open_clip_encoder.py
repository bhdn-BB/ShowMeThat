import os
import shutil
from typing import List
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import euclidean_distances
from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
from PIL.ImageFile import ImageFile
from pydantic import HttpUrl
from tqdm import tqdm
from search_service.config import Config
from search_service.scripts.video_processing import save_frames_from_video, build_youtube_link_from_filename

logger = Config.get_logger(__name__)


class OpenCLIPEncoder:
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model.eval()

    def encode_image_path(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def extract_and_encode_frames(
            self,
            video_urls: List[HttpUrl] | HttpUrl,
            frame_interval_sec: int
    ) -> list[str | None] | None:

        if os.path.exists(Config.OUTPUT_DIR):
            shutil.rmtree(Config.OUTPUT_DIR)
        os.makedirs(Config.OUTPUT_DIR)
        logger.info(f'Created clean folder: {Config.OUTPUT_DIR}')

        image_paths = []
        for url in video_urls:
            logger.info(f"Saving frames from url: {url} (type {type(url)})")
            current_path = save_frames_from_video(url, frame_interval_sec)
            image_paths.extend(current_path)

        embedded_frames = []
        logger.info('Encoding extracted images into embeddings...')

        for filename in tqdm(os.listdir(Config.OUTPUT_DIR), desc="Encoding images"):
            if filename.endswith(Config.IMAGE_FORMATS):
                image_path = os.path.join(Config.OUTPUT_DIR, filename)
                features = self.encode_image_path(image_path)
                embedded_frames.append(features.cpu().numpy())

        if embedded_frames:
            stacked_embeddings = np.vstack(embedded_frames)
            np.save('embedded_frames.npy', stacked_embeddings)
            logger.info('Saved image embeddings to embedded_frames.npy')
            return image_paths
        else:
            logger.warning('No embeddings were generated.')
            return None

    def predict(self, image_features, text_features):
        logits = 100.0 * image_features @ text_features.T
        return logits.squeeze()

    def predict_euclidean(self, image_features, text_features):
        distances = euclidean_distances(text_features.reshape(1, -1), image_features)
        distances = torch.Tensor(distances)
        return distances[0]

    def predict_complex(
            self,
            image_features: torch.Tensor,
            text_features: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            cosine = self.predict(image_features, text_features)
            diff = self.predict_euclidean(image_features, text_features)
            scores = cosine + diff
        return scores

    def get_best_images_by_score(
            self,
            image_file_paths: List[str],
            prompt: str,
            num_top_images: int
    ) -> tuple[list[ImageFile], list[HttpUrl | None] | HttpUrl] | None:

        image_features = np.load('embedded_frames.npy')
        image_features = torch.from_numpy(image_features).to(self.device)
        text_features = self.encode_text(prompt).to(self.device)
        relevance_probs = self.predict_complex(image_features, text_features)
        # relevance_probs = torch.from_numpy(relevance_probs).to(self.device)
        top_indices = torch.argsort(relevance_probs, descending=True)[:num_top_images]

        best_images = []
        youtube_links = []

        for i in top_indices:
            image_path = image_file_paths[i]
            try:
                image = Image.open(image_path)
                best_images.append(image)
                filename = os.path.basename(image_path)
                link = build_youtube_link_from_filename(filename)
                youtube_links.append(link)
            except Exception as e:
                logger.warning(f"Failed to process image '{image_path}': {e}")

        return best_images, youtube_links