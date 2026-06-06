from __future__ import annotations
try:
    from .cfg import CFG, Instruction, BasicBlock, PlaceholderInstr, InstructionBase, Dtype
    from .op_info import resolve_arg_type, OpArgTypeResolveFailure
except ImportError:
    from cfg import CFG, Instruction, BasicBlock, PlaceholderInstr, InstructionBase, Dtype
    from op_info import resolve_arg_type, OpArgTypeResolveFailure


from dataclasses import dataclass

from collections import defaultdict

class MultiDefMap(defaultdict):
    def add_instrs(self, instrs: MultiDefMap) -> None:
        for var_name in instrs:
            for bb_name in instrs[var_name]:
                self[var_name][bb_name] = instrs[var_name][bb_name]

    def extend_key(self, key: str, from_map: MultiDefMap) -> None:
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
    uses: MultiDefMap | None  = None

    # The set of all definitions that are defined within the block
    # that are not killed
    defsout: dict[str, Instruction] =  None
    
    # The set of all definitions that define variables killed 
    # by other definitions within the block 
    killed: MultiDefMap | None = None 
    
    # The set of all definitions from all blocks that can 
    # reach the end of the current basic block. These are defs 
    # accessible to other basic blocks
    exit_reaches: MultiDefMap | None = None 
    
    @classmethod
    def create(cls, bb_label: str, uses: MultiDefMap, defsout: dict[str, Instruction], killed: MultiDefMap, predecessor_reaches: MultiDefMap):
        # defs from predecessors that reach the BB and are not killed 
        # note that we keep the origin of the basic block that defined 
        # the instruction - as given by the defsout inclusion
        reaches = {var_name : predecessor_reaches[var_name] for var_name in  (predecessor_reaches.keys() - killed.keys())}
        for var_name in defsout:
            reaches[var_name] = {bb_label: defsout[var_name]}
        return cls(uses=uses, defsout=defsout, killed=killed, exit_reaches=reaches)
            
def add_undef_instructions(cfg: CFG, dataflow_info: dict[str, BBDataFlowInfo], undef_map: set[ str ]) -> CFG:
    # We need to resolve the dtype of the variables in the undef_map, which is typed 
    # as a set but should really be a map. The types can be resolved by 
    # inspecting the operation that is using the variable.
    # This path also creates problems because a variable may be input 
    # to different operations
    raise NotImplementedError

    

def run_dataflow_analysis(cfg: CFG) -> dict[str, BBDataFlowInfo]:
    cfg_predecessor_map = cfg.get_predecessor_map()
    root_bb_label = cfg.get_root_block().label
    
    def do_dataflow_analysis(bb_label: str, predecessor_reaches: MultiDefMap, undef_map: MultiDefMap, resolve_uses_args: bool = False):
        bb = cfg.label_to_bb_map[bb_label]
        bb_defs : dict[str, Instruction] = {}
        uses_defs = MultiDefMap.create() 
        bb_killed_defs = MultiDefMap.create()
        for instr in bb.instrs:
            if hasattr(instr, "args"): 
                if instr.args is not None:
                    new_args: list[Instruction | PlaceholderInstr] = []
                    for arg in enumerate(instr.args):
                        arg_dest = arg.dest 
                        arg_has_undef_path = False
                        if arg_dest in bb_defs:
                            # Update PlaceHolderInstr with actual instructions
                            if isinstance(arg, PlaceholderInstr):
                                new_args.append(bb_defs[arg_dest])
                            else:
                                new_args.append(arg)
                            continue 
                        else:
                            # Check if a variable is defined prior to this basic block
                            if arg_dest in predecessor_reaches:
                                arg_has_undef_path = not all(
                                    arg_dest in dataflow_map[predecessor_bb_label].exit_reaches 
                                    for predecessor_bb_label in cfg_predecessor_map[bb_label]
                                )
                                # If there is a single instruction defining a variable from predecessors
                                # and all predecessor basic blocks have that same reaching instruction
                                # we can disambiguate a PlaceHolderInstr
                                # Otherwise we leave it unresolved
                                if resolve_uses_args and not arg_has_undef_path and len(predecessor_reaches[arg_dest]) == 1 and isinstance(arg, PlaceholderInstr):
                                    new_args.append(list(predecessor_reaches[arg_dest].values())[0])
                                else:
                                    new_args.append(arg)
                                uses_defs.extend_key(arg_dest, predecessor_reaches)
                            else:
                                arg_has_undef_path = True
                                new_args.append(arg)

                            # If there is an undef path, just insert a use from the root block 
                            # of this variable. The caller can then choose what to do.
                            if arg_has_undef_path:
                                maybe_resolved_dtype = resolve_arg_type(arg_name=arg_dest, instr=instr)
                                predecessor_dtypes_for_arg = set([instr.dtype for bb, instr in predecessor_reaches[arg_dest]])
                                if isinstance(maybe_resolved_dtype, OpArgTypeResolveFailure):
                                    # User error, instr should not receive arguments
                                    if maybe_resolved_dtype == OpArgTypeResolveFailure.NotRequired:
                                        # Just use any type
                                        maybe_resolved_dtype = int
                                    # E.g. for print ops 
                                    if maybe_resolved_dtype == OpArgTypeResolveFailure.RequiresDefinition:
                                        # Use the same type as another reaches value
                                        if len(predecessor_reaches[arg_dest]) >= 1:
                                            maybe_resolved_dtype = next(iter(predecessor_reaches[arg_dest].values())).dtype
                                        # Unbacked - just use any type
                                        else:
                                            maybe_resolved_dtype = int 
                                
                                # According to to_ssa from examples, Bril spec requires a single dtype 
                                # for each variable in a function, so we will just insert the undef with 
                                # whichever dtype is most recent in the program so far. After all, 
                                # undef instructions create dead links anyway 
                                undef_map[arg_dest] = maybe_resolved_dtype
                                uses_defs.extend_key(arg_dest, {root_bb_label: PlaceholderInstr.create(arg_dest, op="undef")})

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

                    assert len(new_args) == len(instr.args)
                    instr.args = new_args 

            if hasattr(instr, "dest") and instr.dest is not None:
                # If the BB overwrites another def within the BB
                if instr.dest in bb_defs:
                    bb_killed_defs[instr.dest][bb_label] = bb_defs.pop(instr.dest)
                # If the current instruction overwrites a def from a predecessor, or an undef 
                # inserted in uses_def, add it to the killed set
                elif instr.dest in predecessor_reaches:
                    bb_killed_defs.extend_key(instr.dest, predecessor_reaches)
                elif instr.dest in uses_defs:
                    bb_killed_defs.extend_key(instr.dest, uses_defs)

                # No other instruction is killed
                bb_defs[instr.dest] = instr

        return BBDataFlowInfo.create(bb_label=bb_label, uses=uses_defs, killed=bb_killed_defs, defsout=bb_defs, predecessor_reaches=predecessor_reaches), undef_map

    # Initialise dataflow info
    dataflow_map: dict[str, BBDataFlowInfo] = {
        bb_label : do_dataflow_analysis(bb_label=bb_label, predecessor_reaches=set(), undef_map=MultiDefMap.create())[0]
        for bb_label in cfg.bb_successor_map.keys()
    }
    
    def run_single_pass(*, resolve_uses_args: bool = False):
        changed = False
        # Track which variables may be undefined 
        undef_map: dict[str, Dtype] = {}
        ready_set: set[str] = set()
        to_process: list[str] = [root_bb_label]
        while to_process:
            bb_label = to_process.pop()
            remaining_predecessors = cfg_predecessor_map[bb_label] - ready_set
            if len(remaining_predecessors) > 0:
                to_process.append(bb_label)
                for rpred in remaining_predecessors:
                    to_process.append(rpred)
            else:
                current_dataflow_info = dataflow_map[bb_label]
                predecessor_reaches = MultiDefMap.create() 
                for predecessor_bb_label in cfg_predecessor_map[bb_label]:
                    predecessor_dataflow_info = dataflow_map[predecessor_bb_label]
                    predecessor_reaches.add_instrs(predecessor_dataflow_info.exit_reaches)
                new_dataflow_info, undef_map = do_dataflow_analysis(
                    bb_label=bb_label, predecessor_reaches=predecessor_reaches,
                    undef_map=undef_map, resolve_uses_args=resolve_uses_args
                )
                if new_dataflow_info != current_dataflow_info:
                    changed = True 
            # Update ready set and dataflow info
            dataflow_map[bb_label] = new_dataflow_info
            ready_set.add(bb_label)
        return changed, undef_map 
            

    # Fixed point algorithm for dataflow analysis
    # 1. Ensure we've done dataflow analysis for all the predecessors before 
    # processing the current basic block
    # 2. After the predecessors are ready, do dataflow analysis for the current block
    # 3. Check if the dataflow info for the current block has changed. If so, we 
    # will need to redo dataflow analysis for the entire CFG
    changed = True
    while changed:
        changed, undef_map = run_single_pass()
    
    # Last run to resolve more PlaceholderInstr
    changed, undef_map = run_single_pass(resolve_uses_args=True)
    assert not changed 
       
    return dataflow_map, undef_map
                        
                

    