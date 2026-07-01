from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    input_data: dict[str, Any]
    output_data: dict[str, Any]
    success: bool
    error_message: str | None = None

class BaseTool:
    tool_name: str
    description: str

    def run(self, input_data: dict[str, Any]) -> ToolResult:
        raise NotImplementedError