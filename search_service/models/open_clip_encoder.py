import os
import shutil
from typing import List
import numpy as np
import torch
from PIL import Image
import open_clip
from tqdm import tqdm
from ShowMeThat.search_service.config_video_processing import ConfigVideoProcessing
from ShowMeThat.search_service.scripts.video_processing import save_frames_from_video

logger = ConfigVideoProcessing.get_logger(__name__)

class OpenCLIPEncoder:
    def __init__(self, model_name='ViT-B-32', pretrained='laion2b_s34b_b79k'):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def encode_image(self, image_path):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.autocast(self.device.type):
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text(self, texts):
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad(), torch.autocast(self.device.type):
            text_features = self.model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def extract_and_encode_frames(
            self,
            video_urls: List[str],
            frame_interval_sec: int =10
    ) -> None:

        if os.path.exists(ConfigVideoProcessing.OUTPUT_DIR):
            shutil.rmtree(ConfigVideoProcessing.OUTPUT_DIR)
        os.makedirs(ConfigVideoProcessing.OUTPUT_DIR)
        logger.info(f'Created clean folder: {ConfigVideoProcessing.OUTPUT_DIR}')

        for video_url in video_urls:
            logger.info(f'Extracting frames from {video_url}')
            save_frames_from_video(video_url, frame_interval_sec)

        embedded_frames = []

        logger.info('Encoding extracted images into embeddings...')
        for filename in tqdm(os.listdir(ConfigVideoProcessing.OUTPUT_DIR), desc="Encoding images"):
            if filename.endswith(ConfigVideoProcessing.IMAGE_FORMATS):
                image_path = os.path.join(ConfigVideoProcessing.OUTPUT_DIR, filename)
                features = self.encode_image(image_path)
                embedded_frames.append(features.cpu().numpy())

        if embedded_frames:
            stacked_embeddings = np.vstack(embedded_frames)
            np.save('embedded_frames.npy', stacked_embeddings)
            logger.info('Saved image embeddings to embedded_frames.npy')
        else:
            logger.warning('No embeddings were generated.')

    def predict(self, image_features, text_features):
        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return logits