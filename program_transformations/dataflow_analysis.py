from __future__ import annotations

from email.policy import default
from typing import cast

try:
    from .cfg import CFG, BasicBlock, Dtype, DtypeType, Instruction, PlaceholderInstr
    from .op_info import OpArgTypeResolveFailure, maybe_resolve_arg_type
except ImportError:
    from cfg import (
        CFG,
        BasicBlock,
        Dtype,
        DtypeType,
        Instruction,
        PlaceholderInstr,
    )
    from op_info import OpArgTypeResolveFailure, maybe_resolve_arg_type


import dataclasses
from collections import defaultdict
from dataclasses import dataclass


class MultiDefMap(defaultdict):
    def add_instrs(self, instrs: MultiDefMap) -> None:
        for var_name in instrs:
            for bb_name in instrs[var_name]:
                self[var_name][bb_name] = instrs[var_name][bb_name]

    def extend_key(self, key: str, from_map: MultiDefMap | dict[str]) -> None:
        for bb_name in from_map[key]:
            self[key][bb_name] = from_map[key][bb_name]

    @classmethod
    def create(cls):
        return cls(dict)


@dataclass
class BBDataFlowInfo:
    # The set of all variables within the block that have
    # no prior definitions within the block. These are uses that
    # are not satisfied within the block
    uses: MultiDefMap = dataclasses.field(default_factory=MultiDefMap.create)

    # The set of all definitions that are defined within the block
    # that are not killed
    defsout: dict[str, Instruction] = dataclasses.field(default_factory=dict)

    # The set of all definitions that define variables killed
    # by other definitions within the block
    killed: MultiDefMap = dataclasses.field(default_factory=MultiDefMap.create)

    # The set of all definitions from all blocks that can
    # reach the end of the current basic block. These are defs
    # accessible to other basic blocks
    exit_reaches: MultiDefMap = dataclasses.field(default_factory=MultiDefMap.create)

    @classmethod
    def create(
        cls,
        bb_label: str,
        uses: MultiDefMap,
        defsout: dict[str, Instruction],
        killed: MultiDefMap,
        predecessor_reaches: MultiDefMap,
    ):
        # defs from predecessors that reach the BB and are not killed
        # note that we keep the origin of the basic block that defined
        # the instruction - as given by the defsout inclusion
        reaches = {
            var_name: predecessor_reaches[var_name]
            for var_name in (predecessor_reaches.keys() - killed.keys())
        }
        for var_name in defsout:
            assert var_name not in reaches, "Def must have killed any reaching def"
            reaches[var_name] = {bb_label: defsout[var_name]}

        return cls(
            uses=uses,
            defsout=defsout,
            killed=killed,
            exit_reaches=reaches,
        )


def run_dataflow_analysis(
    cfg: CFG,
) -> tuple[dict[str, BBDataFlowInfo], dict[str, Instruction]]:
    cfg_predecessor_map = cfg.get_predecessor_map()
    root_bb_label: str = cfg.get_root_block().label

    def do_dataflow_analysis(
        bb_label: str,
        predecessor_reaches: MultiDefMap,
        undef_map: dict[str, Instruction],
    ):
        bb = cfg.label_to_bb_map[bb_label]
        bb_defs: dict[str, Instruction] = {}
        uses_defs = MultiDefMap.create()
        bb_killed_defs = MultiDefMap.create()

        def try_resolve_undef_dtype(arg, instr: Instruction) -> DtypeType:
            """
            If a PlaceholderInstr has a path from the root block to its use,
            it means there is a path in the program in which a variable is
            not defined. Try to resolve the dtype of the variable here as the
            caller may want to insert undef instructions e.g. when converting
            to SSA
            """

            assert isinstance(arg, PlaceholderInstr), (
                "Resolution is only required for args that have multiple reaching definitions"
            )
            assert arg.dest is not None

            maybe_resolved_dtype = maybe_resolve_arg_type(
                arg_name=arg.dest, instr=instr
            )

            undef_dtype = maybe_resolved_dtype

            if isinstance(maybe_resolved_dtype, OpArgTypeResolveFailure):
                # User error, instr should not receive arguments
                if maybe_resolved_dtype == OpArgTypeResolveFailure.NotRequired:
                    # Just use any type
                    undef_dtype = int
                # E.g. for print ops
                if maybe_resolved_dtype == OpArgTypeResolveFailure.RequiresDefinition:
                    # Use the same type as another reaches value if its available
                    # According to to_ssa from examples, Bril spec requires a single dtype
                    # for each variable in a function, so we will just insert the undef with
                    # whichever dtype is most recent in the program so far. After all,
                    # undef instructions create dead links anyway
                    if len(predecessor_reaches[arg.dest]) >= 1:
                        undef_dtype = next(
                            iter(predecessor_reaches[arg_dest].values())
                        ).dtype
                    # Unbacked - just use any type
                    else:
                        undef_dtype = int

            # --- Previously I tried to handle more complex cases but its
            # unneccesary for now

            # undef_bbs = cfg_predecessor_map[bb_label] - predecessor_reaches[arg_dest].keys()
            # if len(predecessor_dtypes_for_arg) > 1:
            #     for ubb in undef_bbs:
            #         undef_map[arg_dest][ubb] = maybe_resolved_dtype
            # # There is a single predecessor dtype for arg_dest
            # # now just check if it matches the dtype already in undef map or not
            # else:
            #     if arg_dest in undef_map:
            #         if root_bb_label in undef_map[arg_dest]:
            #             if undef_map[arg_dest][root_bb_label] == maybe_resolved_dtype:
            #                 uses_defs.extend_key(arg_dest, {root_bb_label: PlaceholderInstr.create(arg_dest, op="undef")})
            #             else:
            #                 for ubb in undef_bbs:
            #                     undef_map[arg_dest][ubb] = maybe_resolved_dtype
            #                     uses_defs.extend_key(arg_dest, {ubb: PlaceholderInstr.create(arg_dest, op="undef")})
            #         else:
            #             undef_map[arg_dest][root_bb_label] = maybe_resolved_dtype
            #             uses_defs.extend_key(arg_dest, {root_bb_label: PlaceholderInstr.create(arg_dest, op="undef")})
            #     else:
            #         undef_map[arg_dest][root_bb_label] = maybe_resolved_dtype
            #         uses_defs.extend_key(arg_dest, {root_bb_label: PlaceholderInstr.create(arg_dest, op="undef")})
            return undef_dtype

        def add_uses_defs_for_arg(arg: Instruction | PlaceholderInstr) -> None:
            """
            Check if arg has an undef path i.e. there is a path from
            the root block to the current block that does not define
            a arg
            """
            assert arg.dest is not None
            arg_has_undef_path: bool = False

            # If arg in bb_defs: nothing to do as there is a single
            # definition that should back this arg if its a placeholder
            if arg.dest not in bb_defs:
                # Check if a variable is defined prior to this basic block
                if arg.dest in predecessor_reaches:
                    arg_has_undef_path = not all(
                        arg.dest in dataflow_map[predecessor_bb_label].exit_reaches
                        for predecessor_bb_label in cfg_predecessor_map[bb_label]
                    )
                    uses_defs.extend_key(arg.dest, predecessor_reaches)
                else:
                    arg_has_undef_path = True

                if arg_has_undef_path:
                    undef_dtype = try_resolve_undef_dtype(arg, instr)
                    assert not isinstance(undef_dtype, Instruction)

                    # Create the root block undef instr that will be referenced
                    # as a use for the current basic block
                    root_block_undef_instr = Instruction.create_undef_instr(
                        arg.dest, undef_dtype
                    )
                    undef_map[arg.dest] = root_block_undef_instr
                    tmp_undef_map = MultiDefMap.create()
                    tmp_undef_map[arg.dest][root_bb_label] = root_block_undef_instr
                    uses_defs.extend_key(arg.dest, tmp_undef_map)

        for instr in bb.instrs:
            # Handle reads
            if instr.args is not None:
                for arg in instr.args:
                    assert arg.dest is not None, (
                        "Instruction args must have destination names"
                    )
                    add_uses_defs_for_arg(arg)

            # Process creation of a new variable
            if instr.dest is not None:
                # TODO: really bb_killed_defs should be str->dict[str, list[Instruction]]
                # since a basic block can overwrite a definition multiple times
                # for now just keep the most recent killed def from the current basic block
                if instr.dest in bb_defs:
                    bb_killed_defs[instr.dest][bb_label] = bb_defs.pop(instr.dest)
                # If there is a previous instruction that read from a variable with the
                # same name, query the uses_defs
                elif instr.dest in uses_defs:
                    bb_killed_defs.extend_key(instr.dest, uses_defs)
                # If there isn't a use in the current BB, reference the predecessor reaches
                elif instr.dest in predecessor_reaches:
                    bb_killed_defs.extend_key(instr.dest, predecessor_reaches)

                # Update the current reaching def
                bb_defs[instr.dest] = instr

        # Now check that each reaching def variable name has a path
        # from the root block that defines a variable with the same destination
        # If not, we need to insert an undef at the beginning of the root block
        for var_name in predecessor_reaches:
            if var_name not in undef_map and not all(
                var_name in dataflow_map[pred_bb_label].exit_reaches
                for pred_bb_label in cfg_predecessor_map[bb_label]
            ):
                # Pick any instruction that defines the variable as Bril requires that each variable carries a single dtype
                var_def = cast(
                    Instruction, list(predecessor_reaches[var_name].values())[0]
                )

                assert not isinstance(var_def.dtype, Instruction)
                assert var_def.dtype is not None
                assert var_def.dest is not None
                root_block_undef_instr = Instruction.create_undef_instr(
                    var_def.dest, var_def.dtype
                )
                undef_map[var_def.dest] = root_block_undef_instr

        return BBDataFlowInfo.create(
            bb_label=bb_label,
            uses=uses_defs,
            killed=bb_killed_defs,
            defsout=bb_defs,
            predecessor_reaches=predecessor_reaches,
        )

    dataflow_map: dict[str, BBDataFlowInfo] = defaultdict(BBDataFlowInfo)

    def run_single_pass(*, resolve_uses_args: bool = False):
        changed = False
        # undef instructions that should be inserted in the root block
        undef_map: dict[str, Instruction] = {}
        # Track which variables may be undefined
        ready_set: set[str] = set()
        current_level_successors: list[str] = [root_bb_label]
        next_level_successors: list[str] = []
        while current_level_successors:
            bb_label = current_level_successors.pop()

            # If we've already visited the block, continue
            if bb_label in ready_set:
                # After we've seen all current level successors,
                # visit the next level - breadth first traversal
                if not current_level_successors:
                    current_level_successors = next_level_successors
                    next_level_successors = []
                continue

            remaining_predecessors = cfg_predecessor_map[bb_label] - ready_set
            if len(remaining_predecessors) > 0:
                # Add the block, then its predecessors
                current_level_successors.append(bb_label)
                for rpred_label in remaining_predecessors:
                    current_level_successors.append(rpred_label)
                continue

            assert len(remaining_predecessors) == 0, (
                "Can only visit a basic block if all its predecessors are ready"
            )

            current_dataflow_info = dataflow_map[bb_label]
            predecessor_reaches = MultiDefMap.create()
            for predecessor_bb_label in cfg_predecessor_map[bb_label]:
                predecessor_dataflow_info = dataflow_map[predecessor_bb_label]
                predecessor_reaches.add_instrs(predecessor_dataflow_info.exit_reaches)
            new_dataflow_info = do_dataflow_analysis(
                bb_label=bb_label,
                predecessor_reaches=predecessor_reaches,
                undef_map=undef_map,
            )
            if new_dataflow_info != current_dataflow_info:
                dataflow_map[bb_label] = new_dataflow_info
                changed = True

            for next_bb_label in cfg.bb_successor_map[bb_label]:
                # If the successor BB is already ready, we don't need to add
                # it to the worklist again. If the current BB has changed, the
                # successor BB will be revisited again on a later pass
                if next_bb_label not in ready_set:
                    next_level_successors.append(next_bb_label)

            # BB has been processed so add to the ready set
            ready_set.add(bb_label)

            # After we've seen all current level successors,
            # visit the next level - breadth first traversal
            if not current_level_successors:
                current_level_successors = next_level_successors
                next_level_successors = []

        return changed, undef_map

    # Fixed point algorithm for dataflow analysis
    # 1. Ensure we've done dataflow analysis for all the predecessors before
    # processing the current basic block
    # 2. After the predecessors are ready (breadth first), do dataflow analysis for the current block
    # 3. Check if the dataflow info for the current block has changed. If so, we
    # will need to redo dataflow analysis for the entire CFG on a later pass
    changed = True
    while changed:
        changed, undef_map = run_single_pass(resolve_uses_args=False)

    return dataflow_map, undef_map
