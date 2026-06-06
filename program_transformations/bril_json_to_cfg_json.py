from ast import arg
from email.policy import default
import sys
import pprint
from pathlib import Path
import argparse
from typing import Any, TypeAlias, Generator, cast
import json
from collections import defaultdict

try:
    from . import cfg, utils
    from .utils import InstructionType, BasicBlockType, CFGType, FnToCFGInfo
except ImportError:
    import cfg
    import utils
    from utils import InstructionType, BasicBlockType, CFGType, FnToCFGInfo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=False)
    args = parser.parse_args()
    if args.path:
        json_path: Path = args.path
        program = json.loads(json_path.read_text())
    else:
        program = json.load(sys.stdin)

    structured_program = cfg.Program.from_bril_dict(program)
    print(json.dumps(structured_program.asdict(), indent=2))

if __name__ == "__main__":
    main()
