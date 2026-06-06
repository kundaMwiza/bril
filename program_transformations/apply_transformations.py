import argparse
import pprint
import copy
from pathlib import Path
import json
from typing import Any
try:
    from . import transformations
except ImportError:
    import transformations

    

def main(args):
    assert args.bril_json.exists()
    original_program = json.loads(args.bril_json.read_text())
    transformations.apply_transformations(original_program, transformations=args.transformations, preset=args.preset)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bril_json", type=Path)
    transformations_group = parser.add_mutually_exclusive_group()
    transformations_group.add_argument("--transformations", type=str, nargs="*", choices=transformations.AVAILABLE_TRANSFORMATIONS.keys(), default=[])
    transformations_group.add_argument("--preset", type=str, choices=transformations.TRANSFORMATION_PRESETS.keys())
    args = parser.parse_args()
    main(args)
