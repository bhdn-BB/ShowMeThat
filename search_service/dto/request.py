from pydantic import BaseModel, HttpUrl, conint
from typing import List

class SearchRequest(BaseModel):
    video_links: List[HttpUrl]
    prompt: str
    quality_processed: conint(ge=1, le=10)
    num_slides: conint(ge=1, le=5)
