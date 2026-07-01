from sqlalchemy.orm import Session

from backend.app.agents.tools.base_tool import BaseTool
from backend.app.agents.tools.calculator_tool import CalculatorTool
from backend.app.agents.tools.document_search_tool import DocumentSearchTool
from backend.app.agents.tools.mock_api_tool import MockApiTool


def build_tool_registry(db: Session) -> dict[str, BaseTool]:
    return {
        "calculator": CalculatorTool(),
        "mock_api": MockApiTool(),
        "document_search": DocumentSearchTool(db=db),
    }