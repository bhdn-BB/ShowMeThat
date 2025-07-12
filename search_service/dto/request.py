from pydantic import BaseModel, HttpUrl, conint
from typing import List

class BaseSearchRequest(BaseModel):
    video_links: List[HttpUrl]
    quality_processed: conint(ge=1, le=10)
    num_slides: conint(ge=1, le=5)


class SearchByTextPrompt(BaseSearchRequest):
    prompt: str
