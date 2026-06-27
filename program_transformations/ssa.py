from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from unittest import mock

try:
    from . import cfg, dataflow_analysis, instruction_effects
    from .cfg import CFG, BasicBlock, Instruction, PlaceholderInstr, Program
    from .dataflow_analysis import BBDataFlowInfo
except ImportError:
    import cfg
    import dataflow_analysis
    import instruction_effects
    from cfg import CFG, BasicBlock, Instruction, PlaceholderInstr, Program
    from dataflow_analysis import BBDataFlowInfo

SSA_VAR_TAG = ".ssa"


def insert_ssa_get_instructions(
    cfg: CFG,
    dataflow_info: dict[str, BBDataFlowInfo],
    dominator_info: CFG.DominanceInfo,
) -> dict[str, dict[str, Instruction]]:
    breakpoint()
    # Algorithm to insert the equivalent of phi instructions in the CFG.
    # In each join node, insert a get instruction - later when variables are renamed
    # the corresponding set instructions will be inserted

    # list[bb_label]
    worklist: list[str] = [cfg.get_root_block().label]
    seen: set[str] = set()

    # bb -> [var_name -> Instruction]
    bb_gets_inserted: dict[str, dict[str, Instruction]] = defaultdict(dict)

    # Depth first preorder traversal
    # TODO: use imdom tree for traversal
    while worklist:
        bb_label = worklist.pop()
        seen.add(bb_label)
        assert bb_label in dataflow_info
        # Handle defs
        for var_name, var_instr in itertools.chain(
            dataflow_info[bb_label].defsout.items(), bb_gets_inserted[bb_label].items()
        ):
            for df_bb in dominator_info.dominance_frontier_map[bb_label]:
                inserted_get_instr = bb_gets_inserted[df_bb].get(var_name, None)

                # BB in DF does not yet have a get instr inserted for var_name
                if inserted_get_instr is None:
                    assert var_instr.dtype is not None
                    inserted_get_instr = Instruction.create_ssa_get_instr(
                        var_name, var_instr.dtype
                    )
                    bb_gets_inserted[df_bb][var_name] = inserted_get_instr

                    # If the BB has a defout with name var_name, no need to add it to
                    # the worklist (it will be handled on first visit)
                    # If it doesn't define the variable, the inserted get instr
                    # now does, so add it to defsout
                    # NB: at this point the dataflow information is no longer valid
                    # as we haven't corrected the exit_reaches and killed defs
                    if (
                        var_name not in dataflow_info[df_bb].defsout
                        and df_bb not in worklist
                    ):
                        worklist.append(df_bb)

                assert inserted_get_instr is not None

        # Add successor bb to the worklist if they haven't already been seen and they're not already
        # on the worklist.
        # If they have been seen then the only way we revisit them is if
        # a get instr gets inserted into the block
        for next_bb_label in cfg.bb_successor_map[bb_label]:
            if next_bb_label not in seen and next_bb_label not in worklist:
                worklist.append(next_bb_label)

    # Actually insert the get instructions in the basic blocks
    for bb_label, vars_to_add in bb_gets_inserted.items():
        prepend_instrs = list(vars_to_add.values())
        cfg.label_to_bb_map[bb_label].insert_instrs(prepend_instrs, idx=0)

    return bb_gets_inserted


class ReachingDef:
    def __init__(
        self, instr: Instruction | None = None, previous: ReachingDef | None = None
    ):
        self.instr = instr
        self.previous = previous


def rename_instructions_to_ssa(
    cfg: CFG,
    dominator_info: CFG.DominanceInfo,
    bb_gets_inserted: dict[str, dict[str, Instruction]],
):
    var_reaching_def: defaultdict[str, ReachingDef] = defaultdict(lambda: ReachingDef())
    var_version_counter: dict[str, int] = Counter()

    # There can be multiple sets in a BB for the different get instrs
    # bb -> (canonical variable name, get instr bb name) -> Instruction
    bb_sets_inserted: dict[str, dict[tuple[str, str], Instruction]] = defaultdict(dict)
    # Keep track of which variables have been renamed. This is used
    # when getting the canonical name of a variable e.g. x.5 -> x
    renamed_bbs: set[str] = set()

    def is_dominated_by(candidate_dominator_instr: Instruction, instr: Instruction):
        # Check if instr is dominated by candidate_dominator_instr
        assert candidate_dominator_instr.basic_block is not None
        assert instr.basic_block is not None
        candidate_dominator_bb = candidate_dominator_instr.basic_block
        instr_bb = instr.basic_block
        # Same basic block
        if candidate_dominator_bb.label == instr_bb.label:
            return candidate_dominator_instr.instr_index <= instr.instr_index
        return (
            candidate_dominator_bb.label
            in dominator_info.bb_to_dominators_map[instr_bb.label]
        )

    def update_reaching_def(arg: Instruction, instr: Instruction):
        assert arg.dest is not None, "instruction args must have a dest"
        reaching_def = var_reaching_def[arg.dest]
        while not (
            reaching_def.instr is None
            or is_dominated_by(
                candidate_dominator_instr=reaching_def.instr, instr=instr
            )
        ):
            assert reaching_def.previous is not None
            reaching_def = reaching_def.previous
        var_reaching_def[arg.dest] = reaching_def

    def update_get_set_instr(
        predecessor_bb: BasicBlock,
        get_instr: Instruction,
        *,
        get_instr_bb_renamed: bool,
    ):
        if get_instr_bb_renamed:
            # If we've renamed this BB, then the get dest is based on the
            # new value
            assert get_instr.dest is not None
            original_var = get_instr.dest.rsplit(SSA_VAR_TAG, maxsplit=1)[0]
        else:
            original_var = get_instr.dest

        assert original_var is not None
        original_var_cur_reaching_instr = var_reaching_def[original_var].instr
        assert original_var_cur_reaching_instr is not None

        # Temporarily patch get_instr basic block to correspond to the current
        # predecessor basic block. This is because get instructions can be
        # interpreted as being processed immediately at the end of the current
        # basic block. This ensures that the reaching def that reaches the
        # get instruction is an instruction that does not necessarily dominate
        # the basic block that the get instr is in, but from a program point
        # of view, it is the _actual_ reaching def
        with mock.patch.object(
            type(get_instr),
            "basic_block",
            new_callable=mock.PropertyMock,
            return_value=predecessor_bb,
        ):
            update_reaching_def(original_var_cur_reaching_instr, get_instr)
            updated_reaching_instr = var_reaching_def[original_var].instr
            assert updated_reaching_instr is not None

        # Now we need to check if a set instr has been inserted in the basic
        # block corresponding to updated_reaching_instr
        assert get_instr.basic_block is not None
        assert updated_reaching_instr.dest is not None
        get_instr_for_set_key = (original_var, get_instr.basic_block.label)
        if (
            get_instr_for_set_key
            not in bb_sets_inserted[updated_reaching_instr.basic_block.label]
        ):
            # We must have renamed the BB we are inserting the set instr into
            # this should be guaranteed because of an imdom preorder traversal
            assert updated_reaching_instr.basic_block.label in renamed_bbs
            bb_sets_inserted[updated_reaching_instr.basic_block.label][
                get_instr_for_set_key
            ] = Instruction.create_ssa_set_instr(
                write_to=get_instr, read_from=updated_reaching_instr
            )

    def imdom_preorder_traversal(curr_bb_label: str):
        breakpoint()
        # 1. Args of get instructions are updated by predecessor basic blocks
        # 2. The current basic block will:
        #   - Update the args of NON-get instructions based on reaching defintions
        #   - Rename the get instruction
        curr_bb = cfg.label_to_bb_map[curr_bb_label]
        for instr in curr_bb.instrs:
            if instr.args is not None:
                # get instructions should not have args
                assert instr.op != "get"
                new_args: list[Instruction] = []
                for a in instr.args:
                    assert a.dest is not None
                    update_reaching_def(a, instr)
                    # For non get instructions, the reaching defs of args must be available
                    # as there must be a single reaching definition for SSA
                    assert var_reaching_def[a.dest].instr is not None
                    assert not isinstance(
                        var_reaching_def[a.dest].instr, PlaceholderInstr
                    )
                    new_args.append(var_reaching_def[a.dest].instr)
                # Update args
                instr.args = new_args

            if instr.dest is not None:
                # For get instructions, reaching defs for instr.dest
                # may _no longer_ dominate the get instructions if the BB
                # they're from does not dominate the current BB i.e.
                # the current BB is in their dominance frontier.
                # So here the reaching def will change to one prior
                update_reaching_def(instr, instr)
                new_dest = f"{instr.dest}{SSA_VAR_TAG}{var_version_counter[instr.dest]}"
                var_version_counter[instr.dest] += 1
                assert new_dest not in var_reaching_def
                # Prior to the definition of this variable, the reaching definition
                # of new_dest will be the previous reaching def, so update
                # instr and previous
                var_reaching_def[new_dest].instr = var_reaching_def[instr.dest].instr
                var_reaching_def[new_dest].previous = var_reaching_def[
                    instr.dest
                ].previous
                # After this newly renamed instruction, the current reaching def for instr.dest
                # is this newly renamed instr. The previous reaching def is based on the previous
                # of this renamed instr
                var_reaching_def[instr.dest].instr = instr
                var_reaching_def[instr.dest].previous = var_reaching_def[new_dest]
                instr.dest = new_dest

        # BB has now been renamed
        renamed_bbs.add(curr_bb_label)

        # Now handle successor BBs that have get instructions
        for succ_bb_label in cfg.bb_successor_map[curr_bb_label]:
            succ_bb_renamed = succ_bb_label in renamed_bbs
            for get_instr in bb_gets_inserted[succ_bb_label].values():
                update_get_set_instr(
                    curr_bb, get_instr, get_instr_bb_renamed=succ_bb_renamed
                )

        # Visit nodes that are immediately dominated by the current basic block
        for next_bb_label in dominator_info.reverse_imdom_map[curr_bb_label]:
            imdom_preorder_traversal(next_bb_label)

    imdom_preorder_traversal(cfg.get_root_block().label)

    # Actually insert the set instructions in the basic blocks
    for bb_label, vars_to_add in bb_sets_inserted.items():
        set_instrs_to_insert = list(vars_to_add.values())
        bb_for_insert = cfg.label_to_bb_map[bb_label]
        last_instr = bb_for_insert.instrs[-1]
        # calls can return values, so sets need to be after the value is returned
        # additionally, the call instruction calls into a different CFG, so the reaching
        # definitions in that CFG will be different
        if last_instr.op in {"jmp", "br", "ret"}:
            bb_for_insert.insert_instrs(set_instrs_to_insert, idx=-1)
        else:
            # append
            bb_for_insert.insert_instrs(
                set_instrs_to_insert, idx=len(bb_for_insert.instrs)
            )


def convert_to_ssa(cfg: CFG) -> CFG:
    dataflow_info, undef_map = dataflow_analysis.run_dataflow_analysis(cfg)
    # If there are undefined variables in the CFG, create
    # create variables in the root block that are initialised with the
    # undef instruction
    if undef_map:
        cfg.get_root_block().insert_instrs(list(undef_map.values()))
        dataflow_info, undef_map = dataflow_analysis.run_dataflow_analysis(cfg)
        assert not undef_map

    # After inserting ssa_instructions, there should not be any placeholder instructions
    dominator_info = cfg.get_dominator_info()
    # First insert get / phi instrs
    bb_gets_inserted = insert_ssa_get_instructions(cfg, dataflow_info, dominator_info)
    # Now rename variables and insert set instructions corresponding
    # to reaching definitions
    rename_instructions_to_ssa(cfg, dominator_info, bb_gets_inserted)

    return cfg
