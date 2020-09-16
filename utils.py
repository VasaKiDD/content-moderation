import argparse
import ast

from yaml import safe_load


def create_parser():
    parser = argparse.ArgumentParser(description="AutoML parameters and stuff")
    parser.add_argument("-i", "--parameters", help="path of parameter file", type=str)
    parser.add_argument(
        "-s", "--spec_checkpoint", help="specific model to resume training", type=str
    )
    return parser


def e_notation(val) -> bool:
    """
    Returns true if the string represented by val is a number written as an exponential in
    python. For example val = 1e4 or val = 1e-4.

    Args:
        val: Value to be parsed

    Returns:
        `True` if `val` is an string representing e notation. Otherwise return `False`.

    """
    if not isinstance(val, str):
        return False
    if "e-" not in val:
        return False
    split = val.split("e-") if "-" in val else val.split("e")
    try:
        first = float(split[0])
        sec = float(split[1])
        return True
    except ValueError:
        return False


def recover_values(values):
    if isinstance(values, str):
        if values[0] == "(" or values[0] == "[" or e_notation(values):
            return ast.literal_eval(values)
    return values


def clean_yaml(yaml):
    clean_dict = {}
    for key, val in yaml.items():
        clean_dict[key] = recover_values(val) if not isinstance(val, dict) else clean_yaml(val)
    return clean_dict


def load_params(path: str):
    with open(path, "rb") as f:
        yaml = safe_load(f)
    return clean_yaml(yaml)
