import os
import shutil
from typing import List
import numpy as np
import torch
from PIL import Image
import open_clip
from PIL.ImageFile import ImageFile
from pydantic import HttpUrl
from tqdm import tqdm
from search_service.config import Config
from search_service.scripts.video_processing import save_frames_from_video, build_youtube_link_from_filename


logger = Config.get_logger(__name__)

class OpenCLIPEncoder:
    def __init__(self, model_name='ViT-B-32', pretrained='laion2b_s34b_b79k'):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            precision="fp32"
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def encode_image_path(self, image_path: str) -> torch.Tensor:
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.autocast(self.device.type, enabled=False):
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text(self, text: str) -> torch.Tensor:
        prompt = f"a photo of {text}"
        tokens = self.tokenizer(prompt).to(self.device)
        with torch.no_grad(), torch.autocast(self.device.type):
            text_features = self.model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def extract_and_encode_frames(
            self,
            video_urls: List[HttpUrl] | HttpUrl,
            frame_interval_sec: int
    ) -> list[list[str] | None] | None:

        if os.path.exists(Config.OUTPUT_DIR):
            shutil.rmtree(Config.OUTPUT_DIR)
        os.makedirs(Config.OUTPUT_DIR)
        logger.info(f'Created clean folder: {Config.OUTPUT_DIR}')

        image_paths = list(list())

        for url in video_urls:
            logger.info(f"Saving frames from url: {url} (type {type(url)})")
            current_path = save_frames_from_video(url, frame_interval_sec)
            image_paths.append(current_path)

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


    @staticmethod
    def predict(image_features, text_features):
        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return logits

    def get_best_images_by_score(
            self,
            image_file_paths: List[str] | List[List[str]],
            prompt: str,
            num_top_images: int
    ) -> tuple[list[ImageFile], list[HttpUrl | None] | HttpUrl] | None:

        if not image_file_paths:
            logger.warning("Empty image file list.")
            return None

        image_features = np.load('embedded_frames.npy')
        image_features = torch.from_numpy(image_features).to(self.device).to(torch.bfloat16)
        text_features = self.encode_text(prompt)

        # if image_features.dtype != text_features.dtype:
        #     image_features = image_features.to(text_features.dtype)
        relevance_probs = self.predict(image_features=image_features, text_features=text_features)[0]
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