import collections
import dataclasses
import typing

try:
    from . import utils
except ImportError:
    import utils

# uses(b): definitions that are not present in b, but are used by instructions in b
# reaches(b): the set of all definitions from all basic blocks that can reach b (including definitions in b)
# killed(b): redefinitions of variables in b that kill previous definitions
# defsout(b): definitions that are defined in b that are not killed in b


@dataclasses.dataclass
class VariableDefinition:
    dest: str
    op: str
    type: str
    value: str
    pos: dict[str, int]


# map from variable name to set[Variable]
DefMap = typing.TypeAlias = dict[str, list[VariableDefinition]]


class BBDataFlowInfo:
    def __init__(self, bb: utils.BasicBlockType):
        self._bb: utils.BasicBlockType = bb
        # These are definitions that are defined in b that are not killed in
        # the basic block
        self.defs_out: dict[str, VariableDefinition] = {}

        # The set of variables that are used by instructions in the
        # basic block but are not defined in the basic block.
        # At the end of TODO: it is required that all uses are satisfied
        # by variables in the reaches map
        self.uses: set[str] = set()

        # These are definitions that are killed by definitions
        # in the basic block.
        self.killed: DefMap = collections.defaultdict(list)

        self.reaches: DefMap = collections.defaultdict(list)

        self._init()

    def _init(self):
        for instr in self._bb:
            # variable definition
            if "dest" in instr:
                if "pos" not in instr:
                    raise RuntimeError(
                        "-p to bril2json is required in order to perform dataflow analysis"
                    )

                v = VariableDefinition(**instr)
                # If we've already seen an earlier definition of a
                # variable in this basic block, then this instruction
                # kills prior definitions
                if v.dest in self.defs_out:
                    self.killed[v.dest].append(self.defs_out[v.dest])
                    self.defs_out[v.dest] = v
                else:
                    self.defs_out[v.dest] = v

            # Add arguments that are not satisfied by definitions
            # in the basic block
            if "args" in instr:
                for arg in instr["args"]:
                    if arg not in self.defs_out:
                        self.uses.add(arg)

        # Initialise the reaches set
        for name, defout in self.defs_out.items():
            self.reaches[name].append(defout)

    def update_reaches(
        self, predecessors: set[str], bb_data_flow_info_map: dict[str, BBDataFlowInfo]
    ):
        raise NotImplementedError
