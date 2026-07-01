import ast
import operator
from typing import Any

from backend.app.agents.tools.base_tool import BaseTool, ToolResult

class CalculatorTool(BaseTool):
    tool_name = "calculator"
    description = "Safely evaluates simple arithmetic expressions."

    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
    }

    def run(self, input_data: dict[str, Any]) -> ToolResult:
        expression = str(input_data.get("expression", "")).strip()

        if not expression:
            return ToolResult(
                tool_name=self.tool_name,
                input_data=input_data,
                output_data={},
                success=False,
                error_message="expression is required",
            )

        try:
            parsed_expression = ast.parse(expression, mode="eval")
            value = self.evaluate_node(parsed_expression.body)

            return ToolResult(
                tool_name=self.tool_name,
                input_data={"expression": expression},
                output_data={
                    "result": value,
                    "result_text": str(round(value, 6))
                    if isinstance(value, float)
                    else str(value),
                },
                success=True,
            )

        except Exception as exc:
            return ToolResult(
                tool_name=self.tool_name,
                input_data={"expression": expression},
                output_data={},
                success=False,
                error_message=f"Invalid arithmetic expression: {exc}",
            )

    def evaluate_node(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value

        if isinstance(node, ast.BinOp):
            operator_type = type(node.op)

            if operator_type not in self.allowed_operators:
                raise ValueError("unsupported operator")

            left_value = self.evaluate_node(node.left)
            right_value = self.evaluate_node(node.right)

            return self.allowed_operators[operator_type](left_value, right_value)

        if isinstance(node, ast.UnaryOp):
            operator_type = type(node.op)

            if operator_type not in self.allowed_operators:
                raise ValueError("unsupported unary operator")

            value = self.evaluate_node(node.operand)

            return self.allowed_operators[operator_type](value)

        raise ValueError("only numbers and arithmetic operators are allowed")