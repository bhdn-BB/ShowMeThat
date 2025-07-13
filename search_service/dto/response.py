from pydantic import BaseModel, HttpUrl
from typing import List

class SearchResponse(BaseModel):
    youtube_links_fragments: List[HttpUrl] | HttpUrl
