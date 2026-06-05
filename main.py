"""Crypto Trading Bot — entry point."""

import argparse
import sys

from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Crypto Trading Bot with AI")
    parser.add_argument(
        "--mode",
        choices=["live", "backtest", "api"],
        default="api",
        help="Run mode (default: api)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=".env",
        help="Path to .env config file",
    )
    args = parser.parse_args()

    logger.info(f"Starting Crypto Trading Bot in {args.mode} mode")

    if args.mode == "live":
        from crypto_trading.bot import TradingBot
        bot = TradingBot(config_path=args.config)
        bot.run()
    elif args.mode == "backtest":
        from crypto_trading.backtest import run_backtest
        run_backtest(config_path=args.config)
    elif args.mode == "api":
        from crypto_trading.api import start_api
        start_api(config_path=args.config)


if __name__ == "__main__":
    main()
