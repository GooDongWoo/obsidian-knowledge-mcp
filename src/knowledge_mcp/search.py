"""Validated search inputs and the filter shared by both retrieval branches."""

from datetime import date, datetime, time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from qdrant_client import models

from .documents import SEOUL


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    embedding_model: Literal["dragonkue/BGE-m3-ko"] = "dragonkue/BGE-m3-ko"
    rerank: bool = False
    document_type: list[str] | None = None
    file_type: list[str] | None = None
    created_from: date | None = None
    created_to: date | None = None
    modified_from: date | None = None
    modified_to: date | None = None
    include_private: bool = False
    limit: int = Field(default=8, ge=1, le=20)

    @model_validator(mode="after")
    def validate_request(self):
        self.query = self.query.strip()
        if not self.query:
            raise ValueError("query must contain text")
        for lower, upper in ((self.created_from, self.created_to), (self.modified_from, self.modified_to)):
            if lower and upper and lower > upper:
                raise ValueError("date range must be ordered")
        return self


class SearchResult(BaseModel):
    point_id: str
    document: str
    source_path: str
    document_type: str
    file_type: str
    security_level: str
    created_at: str
    modified_at: str
    heading_path: list[str] = Field(default_factory=list)
    start_line: int | None = None
    end_line: int | None = None
    page: int | None = None
    paragraph_index: int | None = None
    score: float = Field(allow_inf_nan=False, description="Qdrant RRF fusion score, or raw CrossEncoder relevance score when rerank=True.")


def search_filter(request: SearchRequest) -> models.Filter:
    conditions = [models.FieldCondition(
        key="metadata.security_level",
        match=models.MatchAny(any=["public", "private"] if request.include_private else ["public"]),
    )]
    for field in ("document_type", "file_type"):
        values = getattr(request, field)
        if values is not None:
            conditions.append(models.FieldCondition(key=f"metadata.{field}", match=models.MatchAny(any=values)))
    for field, lower, upper in (
        ("created_at", request.created_from, request.created_to),
        ("modified_at", request.modified_from, request.modified_to),
    ):
        if lower or upper:
            # Date-only upper bounds include the entire Seoul calendar day.
            conditions.append(models.FieldCondition(key=f"metadata.{field}", range=models.DatetimeRange(
                gte=datetime.combine(lower, time.min, SEOUL) if lower else None,
                lte=datetime.combine(upper, time.max, SEOUL) if upper else None,
            )))
    return models.Filter(must=conditions)
