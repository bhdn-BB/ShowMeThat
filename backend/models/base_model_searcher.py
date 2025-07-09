from pydantic import BaseModel, Field, HttpUrl
from typing import List

class BaseModelSearcher(BaseModel):
    search_type: str = Field(..., description="Search type for the model")
    quality_processing: int = Field(..., ge=1, le=10, description="Processing quality from 1 to 10")
    video_links: List[HttpUrl] = Field(..., min_items=1, max_items=5, description="Number of video links from 1 to 5")
    fragment_count: int = Field(..., ge=0, le=10, description="Number of fragments to search (0 to 10)")