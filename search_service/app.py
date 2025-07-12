from fastapi import FastAPI
from search_service.model.open_clip_encoder import OpenCLIPEncoder
from search_service.config import Config
from search_service.dto.request import SearchByTextPrompt
from search_service.dto.response import SearchResponse
import os

app = FastAPI(title="Video Frame Search API")

encoder = OpenCLIPEncoder()

@app.post("/search", response_model=SearchResponse)
def search_frames(request: SearchByTextPrompt):

    frame_interval_sec = 11 - request.quality_processed

    encoder.extract_and_encode_frames(
        video_urls=request.video_links,
        frame_interval_sec=frame_interval_sec
    )

    image_files = [os.path.join(Config.OUTPUT_DIR, image_path)
                   for image_path in os.listdir(Config.OUTPUT_DIR)
                   if image_path.endswith(tuple(Config.IMAGE_FORMATS))]

    best_images, youtube_links = encoder.get_best_images_by_score(
        image_file_paths=image_files,
        prompt=request.prompt,
        num_top_images=request.num_slides
    )
    return SearchResponse(youtube_links_fragments=youtube_links)
