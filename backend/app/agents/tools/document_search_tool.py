from typing import Any
from sqlalchemy.orm import Session

from backend.app.agents.tools.base_tool import BaseTool, ToolResult
from backend.app.retrieval.retrieval_service import RetrievalError, retrieve_chunks

class DocumentSearchTool(BaseTool):
    tool_name = "document_search"
    description = "Searches indexed documents using dense, BM25, or hybrid retrieval."

    def __init__(self, db: Session):
        self.db = db

    def run(self, input_data: dict[str, Any]) -> ToolResult:
        query = str(input_data.get("query", "")).strip()
        document_id = input_data.get("document_id")
        retrieval_provider = str(input_data.get("retrieval_provider", "hybrid")).strip()
        top_k = int(input_data.get("top_k", 3))
        rerank = bool(input_data.get("rerank", True))
        candidate_multiplier = int(input_data.get("candidate_multiplier", 3))

        if not query:
            return ToolResult(
                tool_name=self.tool_name,
                input_data=input_data,
                output_data={},
                success=False,
                error_message="query is required",
            )

        try:
            result = retrieve_chunks(
                db=self.db,
                query=query,
                top_k=top_k,
                document_id=document_id,
                method=retrieval_provider,
                rerank=rerank,
                candidate_multiplier=candidate_multiplier,
            )

            return ToolResult(
                tool_name=self.tool_name,
                input_data={
                    "query": query,
                    "document_id": document_id,
                    "retrieval_provider": retrieval_provider,
                    "top_k": top_k,
                    "rerank": rerank,
                    "candidate_multiplier": candidate_multiplier,
                },
                output_data={
                    "retrieval_method": result.retrieval_method,
                    "reranker_used": result.reranker_used,
                    "reranker_name": result.reranker_name,
                    "retrieved_count": len(result.chunks),
                    "chunks": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "document_id": chunk.document_id,
                            "score": chunk.score,
                            "text": chunk.text,
                            "metadata": chunk.metadata,
                        }
                        for chunk in result.chunks
                    ],
                },
                success=True,
            )

        except RetrievalError as exc:
            return ToolResult(
                tool_name=self.tool_name,
                input_data=input_data,
                output_data={},
                success=False,
                error_message=str(exc),
            )