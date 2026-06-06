from __future__ import annotations

import dataclasses
import functools
import itertools

try:
    from . import instruction_effects, utils
except ImportError:
    import instruction_effects
    import utils

import enum
import json
from collections import defaultdict
from dataclasses import InitVar, field
from typing import Any, Generator, Literal, Optional, TypeAlias, cast


# This is intentionally not a dataclass
# so that we can use dataclasses.asdict when converting
# from the internal IR back to BRIL syntax
# Otherwise we'd end up with a {'dtype': <value>}
# value instead of {'ptr': <value>}
class PtrType:
    def __init__(self, dtype: DtypeType):
        self.dtype = dtype

    def __repr__(self):
        return f"PtrType(dtype={self.dtype})"

    @classmethod
    def from_bril_dtype(cls, ptr_type: dict[str, str]):
        assert "ptr" in ptr_type
        elem_type = ptr_type["ptr"]
        # Handle ptr -> ptr types
        if isinstance(elem_type, dict):
            dtype = cls.from_bril_dtype(elem_type)
        else:
            dtype = from_bril_dtype(elem_type)
        return cls(dtype=dtype)

    def to_bril_dtype(self):
        return {
            "ptr": self.to_bril_dtype(self.dtype)
            if isinstance(self.dtype, PtrType)
            else to_bril_dtype(self.dtype)
        }


Dtype: TypeAlias = int | float | bool | str | PtrType
DtypeType: TypeAlias = type[Dtype]

BrilInstructionType: TypeAlias = dict[str, Any]

TERMINATORS: set[str] = {"jmp", "br", "ret"}
GET_ARGUMENT_OP = "get_argument"

INTERNAL_BB_PREFIX = "__bb"


@functools.lru_cache
def from_bril_dtype(dtype: str) -> DtypeType:
    return eval(dtype)


@functools.lru_cache
def to_bril_dtype(dtype: DtypeType) -> str:
    return dtype.__name__


def replace_type_to_dtype(map: dict[str, Any]):
    if (dtype := map.pop("type", None)) and dtype is not None:
        if isinstance(dtype, dict):
            d = PtrType.from_bril_dtype(dtype)
        elif isinstance(dtype, PtrType):
            d = dtype
        else:
            d = from_bril_dtype(dtype)
        map["dtype"] = d


def replace_dtype_to_type(map: dict[str, Any]):
    if (dtype := map.pop("dtype", None)) and dtype is not None:
        if isinstance(dtype, PtrType):
            d = dtype.to_bril_dtype()
        elif isinstance(dtype, dict) and "ptr" in dtype:
            d = dtype
        else:
            d = to_bril_dtype(dtype)
        map["type"] = d


@dataclasses.dataclass
class InstructionBase:
    op: str
    effects: instruction_effects.Effects
    instr_index: int = field(default=-1, kw_only=True)
    pos: dict[str, int] | None = field(default=None, kw_only=True)

    def __post_init__(self):
        self._basic_block: None | BasicBlock = None

    @property
    def basic_block(self) -> BasicBlock | None:
        return self._basic_block

    @basic_block.setter
    def basic_block(self, new_basic_block: BasicBlock) -> None:
        self._basic_block = new_basic_block

    def to_bril_dict(self) -> BrilInstructionType:
        result = dataclasses.asdict(self)
        replace_dtype_to_type(result)
        result.pop("effects")
        args = result.pop("args", None)
        if args is not None:
            assert isinstance(args, list)
            assert len(args) > 0
            assert isinstance(args[0], (Instruction, PlaceholderInstr))
            result["args"] = [arg.dest for arg in args]
        # Remove optional values
        result = {k: v for k, v in result.items() if v is not None}
        return result

    @classmethod
    def from_bril_dict(cls, kwargs: dict[str, Any]) -> InstructionBase:
        op = kwargs["op"]
        replace_type_to_dtype(kwargs)
        effects = instruction_effects.Effects()
        if op == "const":
            return ConstInstr(**kwargs, effects=effects)
        if op == GET_ARGUMENT_OP:
            # `GET_ARGUMENT_OP` instructions should not be removed or reordered
            # because we need to preserve the CFG signature
            effects.add_write(instruction_effects.World)
            return GetArgumentInstr(**kwargs, effects=effects)

        if op in {"jmp", "br", "call", "ret"}:
            effects.add_write(instruction_effects.Control)
        if op == "print":
            effects.add_write(instruction_effects.IO)

        if op == "load":
            effects.add_read(instruction_effects.Heap)
        elif op in {"store", "alloc"}:
            effects.add_write(instruction_effects.Heap)
        elif op == "free":
            effects.add_write(instruction_effects.Heap)

        args = kwargs.pop("args", None)
        if args is not None:
            args = [PlaceholderInstr.create(arg) for arg in args]

        return Instruction(**kwargs, effects=effects, args=args)

    @classmethod
    def create_undef_instr(cls, arg_name: str, arg_dtype: Dtype) -> InstructionBase:
        return Instruction(
            op="undef",
            effects=instruction_effects.Effects(),
            dest=arg_name,
            dtype=arg_dtype,
        )

    from_dict = from_bril_dict


@dataclasses.dataclass
class PlaceholderInstr(InstructionBase):
    """
    Placeholder instruction awaiting resolution of
    the exact instruction from either the same basic block
    or from a preceeding basic block. This is used to
    initialise the `args` of an instruction until they're resolved
    when converting to SSA.
    """

    dest: str | None = None

    @classmethod
    def create(cls, dest: str, op: str = "placeholder") -> PlaceholderInstr:
        return PlaceholderInstr(
            op=op, effects=instruction_effects.Effects(), instr_index=-1, dest=dest
        )


@dataclasses.dataclass
class ConstInstr(InstructionBase):
    dest: str
    dtype: DtypeType
    value: Dtype


@dataclasses.dataclass
class GetArgumentInstr(InstructionBase):
    dest: str
    dtype: DtypeType
    index: int


@dataclasses.dataclass
class Instruction(InstructionBase):
    """
    Models value and effect instructions
    """

    dest: str | None = None
    dtype: Dtype | None = None
    args: list[Instruction | PlaceholderInstr] | None = None
    funcs: list[str] | None = None
    labels: list[str] | None = None

    def __post_init__(self):
        if self.dest:
            assert self.dtype, (
                "If the destination is provided, the dtype must also be provided"
            )
        if self.dtype:
            assert self.dest, "If the dtype is provided, the dest must also be provided"


@dataclasses.dataclass
class BasicBlock:
    label: str
    bb_index: int
    instrs: list[InstructionBase]

    def __post_init__(self):
        self._cfg = None
        # Have instructions aware which BB they are part of
        for idx, instr in enumerate(self.instrs):
            instr.basic_block = self
            assert instr.instr_index == -1
            instr.instr_index = idx

    @property
    def cfg(self):
        return self._cfg

    @cfg.setter
    def cfg(self, cfg: CFG):
        assert isinstance(cfg, CFG)
        assert self._cfg is None, "CFG can only be set once"
        self._cfg = cfg

    @classmethod
    def from_dict(cls, kwargs: dict[str, Any]):
        # TODO: recovering instruction from this path is broken as we should be using
        # BB index and instruction index to locate which basic block and instruction
        # to use
        label = kwargs["label"]
        instrs = [InstructionBase.from_dict(i) for i in kwargs["instrs"]]
        bb_index = kwargs["bb_index"]
        return BasicBlock(label, instrs, bb_index)

    @classmethod
    def from_bril_list(
        cls, label: str, instrs: list[BrilInstructionType], bb_index: int
    ):
        return BasicBlock(
            label=label,
            instrs=[InstructionBase.from_bril_dict(instr) for instr in instrs],
            bb_index=bb_index,
        )

    def to_bril_list(self) -> list[BrilInstructionType]:
        return [{"label": self.label}, *[i.to_bril_dict() for i in self.instrs]]

    def insert_instrs(self, instrs: list[InstructionBase], idx: int = 0):
        assert idx >= 0
        new_instrs = self.instrs[:idx]
        cur_idx = idx
        for i in itertools.chain(instrs, self.instrs[idx:]):
            i.instr_index = cur_idx
            i.basic_block = self
            new_instrs.append(i)
            cur_idx += 1
        self.instrs = new_instrs


@dataclasses.dataclass
class CFG:
    name: str
    label_to_bb_map: dict[str, BasicBlock]
    bb_successor_map: dict[str, list[str]]
    pos: Optional[dict[str, int]] = None
    dtype: DtypeType | None = None

    @dataclasses.dataclass
    class DominanceInfo:
        bb_to_dominators_map: dict[str, set[str]]
        imdom_map: dict[str, None | str]
        dominance_frontier_map: dict[str, set]

    def __post_init__(self):
        self._program = None
        for bb in self.label_to_bb_map.values():
            bb.cfg = self

    @property
    def program(self):
        return self._program

    @program.setter
    def program(self, program: Program):
        assert isinstance(program, Program)
        assert self._program is None, "Program can only be set once"
        self._program = program

    @classmethod
    def from_dict(cls, kwargs: dict[str, Any]):
        name = kwargs["name"]
        label_to_bb_map = {
            label: BasicBlock.from_dict(bb)
            for label, bb in kwargs["label_to_bb_map"].items()
        }
        bb_successor_map = kwargs["bb_successor_map"]
        pos = kwargs.get("pos")
        return CFG(name, label_to_bb_map, bb_successor_map, pos)

    @classmethod
    def from_bril_dict(
        cls,
        name: str,
        instrs: list[BrilInstructionType],
        args: list[dict[str, str]] = [],
        dtype: DtypeType | None = None,
        pos: Optional[dict[str, int]] = None,
    ) -> CFG:
        label_to_bb: dict[str, BasicBlock] = {}

        # Represent function arguments as instruction as well
        # to simplify things. They are free and will be removed when converting back to
        # canonical Bril SSA
        arg_instrs = [
            {
                "op": GET_ARGUMENT_OP,
                "index": i,
                "dest": arg["name"],
                "type": arg["type"],
            }
            for i, arg in enumerate(args)
        ]
        instrs = arg_instrs + instrs

        for i, basic_block in enumerate(cls._get_basic_block(instrs)):
            maybe_label_instr = basic_block[0]
            if "label" in maybe_label_instr:
                label = maybe_label_instr["label"]
                del basic_block[0]
            else:
                label = f"{INTERNAL_BB_PREFIX}{i}"
            assert label not in label_to_bb
            label_to_bb[label] = BasicBlock.from_bril_list(
                label, basic_block, bb_index=i
            )
            # label_to_bb[label] = basic_block

        cfg: dict[str, list[str]] = defaultdict(list)
        ordered_labels = list(label_to_bb.keys())
        for i, (label, bb) in enumerate(label_to_bb.items()):
            last_instr: InstructionBase = bb.instrs[-1]
            if last_instr.op in ("jmp", "br"):
                assert hasattr(last_instr, "labels")
                last_instr = cast(Instruction, last_instr)
                assert last_instr.labels is not None
                for succ_label in last_instr.labels:
                    cfg[label].append(succ_label)
            elif last_instr.op == "ret":
                # No successors
                cfg[label] = []
            else:
                # Fallthrough to next block if it exists
                cfg[label] = []
                if i + 1 < len(label_to_bb):
                    next_label = ordered_labels[i + 1]
                    cfg[label].append(next_label)

        return CFG(name, label_to_bb, cfg, pos=pos, dtype=dtype)

    def to_bril_dict(self):
        # TODO: will looking at the CFG help here?
        out: dict[str, Any] = {"name": self.name}

        fn_current_index = 0
        fn_args = []

        def remove_additional_instrs_from_bb(bb: list[dict[str, str | int]]) -> None:
            """
            Remove GET_ARGUMENT_OP and additional labels insterted
            when converting from BRIL json to our IR
            """

            nonlocal fn_current_index
            out_bb = []
            for i, instr in enumerate(bb):
                if "op" in instr and instr["op"] == GET_ARGUMENT_OP:
                    assert instr["index"] == fn_current_index
                    fn_args.append({"name": instr["dest"], "type": instr["type"]})
                    fn_current_index += 1
                elif "label" in instr and instr["label"].startswith(INTERNAL_BB_PREFIX):
                    pass
                else:
                    out_bb.append(instr)
            return out_bb

        bb_info = []
        for bb in self.label_to_bb_map.values():
            bb_as_bril_list = bb.to_bril_list()
            bb_as_bril_list_wout_args = remove_additional_instrs_from_bb(
                bb_as_bril_list
            )
            bb_info.extend(bb_as_bril_list_wout_args)

        out["instrs"] = bb_info
        out["args"] = fn_args
        if self.pos is not None:
            out["pos"] = self.pos
        if self.dtype is not None:
            out["dtype"] = self.dtype
        replace_dtype_to_type(out)
        return out

    @staticmethod
    def _get_basic_block(
        instructions: list[BrilInstructionType],
    ) -> Generator[list[BrilInstructionType], None, None]:
        current_block: list[BrilInstructionType] = []
        for instr in instructions:
            # Start of a new basic block
            if "label" in instr and "op" not in instr:
                if current_block:
                    yield current_block
                current_block = [instr]
            elif instr["op"] in TERMINATORS:
                current_block.append(instr)
                yield current_block
                current_block = []
            else:
                current_block.append(instr)

        if current_block:
            yield current_block

    def get_root_block(self) -> BasicBlock:
        # The root block is always the first basic block
        return next(iter(self.label_to_bb_map.values()))

    def get_predecessor_map(self) -> dict[str, set[str]]:
        bb_predecessor_map: dict[str, set[str]] = defaultdict(set)
        for bb, successors in self.bb_successor_map.items():
            for s in successors:
                bb_predecessor_map[s].add(bb)
        return bb_predecessor_map

    def get_dominator_info(self) -> DominanceInfo:
        """
        Get the following information:

        1. BB that dominate each BB
        2. BB that immediately dominate a given BB
        3. The dominance frontier of each BB
        """

        bb_to_predecessors: dict[str, set[str]] = self.get_predecessor_map()

        # Initialise dominators. The root block is only dominated by itself
        bb_to_dominators: dict[str, set[str]] = {
            bb: self.label_to_bb_map.keys() for bb in self.label_to_bb_map
        }
        root_block = self.get_root_block()
        bb_to_dominators[root_block.label] = {root_block.label}

        # Fix point algorithm to determine each basic blocks dominators
        changed = True
        pass_no = 0
        while changed:
            changed = False
            for bb_label in self.label_to_bb_map:
                current_dominators = bb_to_dominators[bb_label]
                new_dominators = set()
                for i, pred_bb in enumerate(bb_to_predecessors[bb_label]):
                    if i == 0:
                        new_dominators |= bb_to_dominators[pred_bb]
                    else:
                        new_dominators &= bb_to_dominators[pred_bb]
                new_dominators.add(bb_label)
                if len(new_dominators) != len(current_dominators) or not all(
                    d in current_dominators for d in new_dominators
                ):
                    changed = True
                    bb_to_dominators[bb_label] = new_dominators

        # bb -> imdom
        imdom_map: dict[str, None | str] = {}
        # imdom -> bb
        reverse_imdom_map: dict[str, set[str]] = defaultdict(set)

        # Get the imdom map
        for bb_label in self.label_to_bb_map:
            # Root block has no immediate dominator
            if bb_label == root_block.label:
                imdom_map[bb_label] = None
                continue
            # Every node dominates itself so exclude it from dom_label
            for dom_label in bb_to_dominators[bb_label] - {bb_label}:
                # the immediate dominator is dominated by all the other
                # dominators
                if all(
                    bb in bb_to_dominators[dom_label]
                    for bb in (bb_to_dominators[bb_label] - {bb_label})
                ):
                    imdom_map[bb_label] = dom_label
                    reverse_imdom_map[dom_label].add(bb_label)
                    break

        # Get the dominance frontier
        ready_set: set[str] = set()
        to_process: list[str] = [root_block.label]
        dominance_frontier_map: dict[str, set[str]] = defaultdict(set)
        while to_process:
            next_bb = to_process[-1]
            # A node is ready to process if its not the imdom of any node or
            # if all nodes its immediately dominates are ready
            if next_bb not in reverse_imdom_map or all(
                bb in ready_set for bb in reverse_imdom_map[next_bb]
            ):
                # Remove the bb since all the nodes it dominates
                # have had their DF calculated
                to_process.pop()

                # Add successors that are not dominated by next_bb
                for successor in self.bb_successor_map[next_bb]:
                    if next_bb not in bb_to_dominators[successor]:
                        dominance_frontier_map[next_bb].add(successor)

                # Add dominance frontier nodes from nodes
                # that next_bb immediately dominates
                for bb in reverse_imdom_map[next_bb]:
                    for df_bb in dominance_frontier_map[bb]:
                        # Note: next_bb == db_bb (self dominance)
                        # so if df_bb (=next_bb) is in the dominance
                        # frontier of a basic block that it immediately
                        # dominates, it needs to be added to the dominance
                        # frontier
                        if next_bb not in bb_to_dominators[df_bb] or next_bb == df_bb:
                            dominance_frontier_map[next_bb].add(df_bb)

                # This node is now ready
                ready_set.add(next_bb)
            else:
                to_process.append(next(iter(reverse_imdom_map[next_bb] - ready_set)))

        if root_block.label in dominance_frontier_map:
            assert len(dominance_frontier_map[root_block.label]) == 0, (
                "The root block must dominate all basic blocks"
            )
        else:
            dominance_frontier_map[root_block.label] = set()

        return self.DominanceInfo(bb_to_dominators, imdom_map, dominance_frontier_map)

    def get_cfg_arguments(self):
        return [
            instr
            for instr in self.get_root_block().instrs
            if isinstance(GetArgumentInstr)
        ]


@dataclasses.dataclass
class Program:
    cfgs: list[CFG]

    def __post_init__(self):
        for cfg in self.cfgs:
            cfg.program = self

    @classmethod
    def from_dict(cls, program: dict[str, Any]) -> Program:
        assert "cfgs" in program
        cfgs: list[CFG] = []
        for cfg in program["cfgs"]:
            cfgs.append(CFG.from_dict(cfg))
        return Program(cfgs)

    @classmethod
    def from_bril_dict(cls, program: dict[str, Any]) -> Program:
        cfgs: list[CFG] = []
        assert "functions" in program, (
            "dict must have functions key with the value being a list of functions"
        )
        for fn in program["functions"]:
            assert isinstance(fn, dict) and "name" in fn and "instrs" in fn, (
                f"Each fn must be of type dict and contain two keys: `name` and `instrs`. Got {fn=}"
            )
            replace_type_to_dtype(fn)
            cfgs.append(CFG.from_bril_dict(**fn))
        return Program(cfgs)

    def asdict(self):
        return dataclasses.asdict(self)

    def to_bril_dict(self) -> dict[str, Any]:
        output_program: dict[str, list[dict[str, str | list[BrilInstructionType]]]] = {
            "functions": []
        }
        for fn_in_cfg_form in self.cfgs:
            output_program["functions"].append(fn_in_cfg_form.to_bril_dict())
        return output_program

    def to_bril_json_str(self) -> str:
        return json.dumps(self.to_bril_dict(), indent=2)
