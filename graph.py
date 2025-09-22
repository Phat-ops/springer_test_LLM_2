from __future__ import annotations
from typing import Any
from langgraph.graph import StateGraph, END

from components.state import StockAgentState
from components.nodes import (
    validate_ticker,
    fetch_stock_data,
    calculate_technical_indicators,
    analyze_market_trend,
    generate_recommendation,
)


def build_graph() -> Any:
    graph = StateGraph(StockAgentState)

    graph.add_node('validate_ticker', validate_ticker)
    graph.add_node('fetch_stock_data', fetch_stock_data)
    graph.add_node('calculate_technical_indicators', calculate_technical_indicators)
    graph.add_node('analyze_market_trend', analyze_market_trend)
    graph.add_node('generate_recommendation', generate_recommendation)

    graph.set_entry_point('validate_ticker')

    def route_after_validate(state: StockAgentState):
        return 'fetch_stock_data' if state.get('is_valid') else 'generate_recommendation'

    graph.add_conditional_edges('validate_ticker', route_after_validate)

    graph.add_edge('fetch_stock_data', 'calculate_technical_indicators')
    graph.add_edge('calculate_technical_indicators', 'analyze_market_trend')
    graph.add_edge('analyze_market_trend', 'generate_recommendation')
    graph.add_edge('generate_recommendation', END)

    return graph.compile()
