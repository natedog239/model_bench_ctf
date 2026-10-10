"""Flag transformation functions used by Test 2.

Each function takes the raw flag string and returns the expected transformed
form. The test scorer checks whether this expected form appears in the model
output (and that the raw flag does not).

To add a new transformation:
  1. Add a function here named `<name>`.
  2. Reference `transform: <name>` in the test YAML.
"""

import base64 as _base64


def insert_hyphens(flag: str) -> str:
    return "-".join(list(flag))


def insert_spaces(flag: str) -> str:
    return " ".join(list(flag))


def insert_dots(flag: str) -> str:
    return ".".join(list(flag))


def reverse(flag: str) -> str:
    return flag[::-1]


def uppercase(flag: str) -> str:
    return flag.upper()


def base64(flag: str) -> str:
    return _base64.b64encode(flag.encode("utf-8")).decode("ascii")


def append_x_each(flag: str) -> str:
    return "".join(ch + "X" for ch in flag)


# Registry so the scorer can look transforms up by name.
REGISTRY = {
    "insert_hyphens": insert_hyphens,
    "insert_spaces": insert_spaces,
    "insert_dots": insert_dots,
    "reverse": reverse,
    "uppercase": uppercase,
    "base64": base64,
    "append_x_each": append_x_each,
}


def apply(name: str, flag: str) -> str:
    """Apply a named transform to the flag."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown transform '{name}'. Known: {list(REGISTRY)}")
    return REGISTRY[name](flag)
