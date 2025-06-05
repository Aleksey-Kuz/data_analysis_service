import logging
import os
from airflow.models import Variable


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


ENV_VARS = {}


def add_variadle(key, val):
    try:
        Variable.set(key, val)
        logging.info(f"Variable {key} set successfully to {val}")
    except Exception:
        logging.exception("Failed to set variable")
        exit(1)


def main():
    for key, val in ENV_VARS.items():
        if val is not None:
            add_variadle(key, val)


if __name__ == "__main__":
    main()
