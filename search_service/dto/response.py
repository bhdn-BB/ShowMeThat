from pydantic import BaseModel
from typing import List

class SearchResponse(BaseModel):
    youtube_links_fragments: List[str]
