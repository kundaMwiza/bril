import json
import sys
from typing import cast
try:
    from .utils import InstructionType, CFGType, BasicBlockType
    from . import cfg
except ImportError:
    from utils import InstructionType, CFGType, BasicBlockType
    import cfg


def main():
    program_in_cfg_form = json.load(sys.stdin)
    print(cfg.Program.from_dict(program_in_cfg_form).to_bril_json_str())

if __name__ == "__main__":
    main()
