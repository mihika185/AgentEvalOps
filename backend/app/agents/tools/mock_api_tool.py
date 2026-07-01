from typing import Any
from backend.app.agents.tools.base_tool import BaseTool, ToolResult

class MockApiTool(BaseTool):
    tool_name = "mock_api"
    description = "Returns deterministic mock API responses for order, customer, and weather queries."

    order_statuses = {
        "ORD-1001": {
            "order_id": "ORD-1001",
            "status": "delivered",
            "delivery_date": "2026-06-26",
            "issue_window_hours": 48,
        },
        "ORD-1002": {
            "order_id": "ORD-1002",
            "status": "in_transit",
            "estimated_delivery": "2026-07-02",
        },
        "ORD-1003": {
            "order_id": "ORD-1003",
            "status": "refund_processing",
            "refund_eta_days": 5,
        },
    }

    customer_plans = {
        "CUST-1001": {
            "customer_id": "CUST-1001",
            "plan": "premium",
            "support_sla_hours": 12,
        },
        "CUST-1002": {
            "customer_id": "CUST-1002",
            "plan": "standard",
            "support_sla_hours": 24,
        },
    }

    weather_by_city = {
        "seattle": {
            "city": "Seattle",
            "condition": "cloudy",
            "temperature_c": 18,
        },
        "sunnyvale": {
            "city": "Sunnyvale",
            "condition": "clear",
            "temperature_c": 24,
        },
        "hyderabad": {
            "city": "Hyderabad",
            "condition": "humid",
            "temperature_c": 29,
        },
    }

    def run(self, input_data: dict[str, Any]) -> ToolResult:
        endpoint = str(input_data.get("endpoint", "")).strip()

        if endpoint == "get_order_status":
            return self.get_order_status(input_data)

        if endpoint == "get_customer_plan":
            return self.get_customer_plan(input_data)

        if endpoint == "get_weather":
            return self.get_weather(input_data)

        return ToolResult(
            tool_name=self.tool_name,
            input_data=input_data,
            output_data={},
            success=False,
            error_message=f"Unsupported mock API endpoint: {endpoint}",
        )

    def get_order_status(self, input_data: dict[str, Any]) -> ToolResult:
        order_id = str(input_data.get("order_id", "")).strip().upper()
        result = self.order_statuses.get(order_id)

        if result is None:
            return ToolResult(
                tool_name=self.tool_name,
                input_data={"endpoint": "get_order_status", "order_id": order_id},
                output_data={},
                success=False,
                error_message=f"No mock order found for order_id '{order_id}'",
            )

        return ToolResult(
            tool_name=self.tool_name,
            input_data={"endpoint": "get_order_status", "order_id": order_id},
            output_data=result,
            success=True,
        )

    def get_customer_plan(self, input_data: dict[str, Any]) -> ToolResult:
        customer_id = str(input_data.get("customer_id", "")).strip().upper()
        result = self.customer_plans.get(customer_id)

        if result is None:
            return ToolResult(
                tool_name=self.tool_name,
                input_data={
                    "endpoint": "get_customer_plan",
                    "customer_id": customer_id,
                },
                output_data={},
                success=False,
                error_message=f"No mock customer found for customer_id '{customer_id}'",
            )

        return ToolResult(
            tool_name=self.tool_name,
            input_data={
                "endpoint": "get_customer_plan",
                "customer_id": customer_id,
            },
            output_data=result,
            success=True,
        )

    def get_weather(self, input_data: dict[str, Any]) -> ToolResult:
        city = str(input_data.get("city", "")).strip().lower()
        result = self.weather_by_city.get(city)

        if result is None:
            return ToolResult(
                tool_name=self.tool_name,
                input_data={"endpoint": "get_weather", "city": city},
                output_data={},
                success=False,
                error_message=f"No mock weather found for city '{city}'",
            )

        return ToolResult(
            tool_name=self.tool_name,
            input_data={"endpoint": "get_weather", "city": city},
            output_data=result,
            success=True,
        )