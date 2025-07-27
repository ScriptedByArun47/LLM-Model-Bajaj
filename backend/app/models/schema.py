from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    query: str

    class Config:
        schema_extra = {
            "example": {
                "query": "Does the policy cover mental health treatment?"
            }
        }

class QueryResponse(BaseModel):
    answer: str
    clause: str
    explanation: str
    tags: List[str]

    class Config:
        schema_extra = {
            "example": {
                "answer": "Yes",
                "clause": "The policy provides coverage for mental health conditions under section 4.2.",
                "explanation": "Mental health is explicitly mentioned as a covered condition in the policy.",
                "tags": ["mental-health", "coverage"]
            }
        }
