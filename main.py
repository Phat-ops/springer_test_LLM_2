from __future__ import annotations
import argparse
from graph import build_graph


def run_agent(ticker: str) -> str:
    graph = build_graph()
    final_state = graph.invoke({'ticker': ticker})
    return final_state.get('report', 'No report generated.')


def main():
    parser = argparse.ArgumentParser(description='Stock Analysis Agent (LangGraph + yfinance)')
    parser.add_argument('tickers', nargs='+', help='Ticker symbols to analyze, e.g., AAPL MSFT GOOGL')
    args = parser.parse_args()

    for t in args.tickers:
        print(run_agent(t))
        print('')


if __name__ == '__main__':
    main()
