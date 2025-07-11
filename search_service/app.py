from fastapi import FastAPI
from models import OpenCLIPEncoder
from config_analyzer.config_analyzer import ConfigAnalyzer as Config
from dto import SearchRequest, SearchResponse

import os

app = FastAPI(title="Video Frame Search API")

encoder = OpenCLIPEncoder()

@app.post("/search", response_model=SearchResponse)
def search_frames(request: SearchRequest):

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
        prompt=request.query,
        num_top_images=request.max_results
    )

    return SearchResponse(video_fragments=youtube_links)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)