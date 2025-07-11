from pydantic import BaseModel
from typing import List

class SearchResponse(BaseModel):
    youtube_fragments: List[str]
    # images: List[str]