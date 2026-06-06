import json
import sys
import dataclasses
from collections import defaultdict
from typing import Optional
try:
    from . import utils
except ImportError:
    import utils



# dominators(b): the set of basic blocks that dominate b
# dominance_frontier(b): the set of basic blocks where b's dominance stops i.e. there is
# a path to a basic block in dominance_frontier(b) that does not pass through b
# dominance_tree: a tree where each node's children are the basic blocks it immediately dominates

@dataclasses.dataclass
class InstructionNode:
    bb: str
    instr: utils.InstructionType

def get_predecessor_map(cfg: utils.CFGType) -> dict[str, set[str]]:
    pred_map: dict[str, set[str]] = defaultdict(set)
    for bb, bb_successors in cfg.items():
        for succ in bb_successors:
            pred_map[succ].add(bb)
    return pred_map

def get_dominators(cfg: utils.CFGType, predecessor_map: dict[str, set[str]]) -> dict[str, set[str]]:
    dominator_map : dict[str, set[str]] = defaultdict(set)
    entry_bb = next(iter(cfg.keys()))
    dominator_map[entry_bb] = {entry_bb}

    changed = True
    while changed:
        changed = False
        for bb in cfg.keys():
            if bb == entry_bb:
                continue
            new_doms: set[str] = set()
            for pred in predecessor_map[bb]:
                new_doms &= dominator_map[pred]
            new_doms.add(bb)
            if new_doms != dominator_map[bb]:
                dominator_map[bb] = new_doms
                changed = True

    return dominator_map

def get_imdom_and_dominator_tree(dominator_map: dict[str, set[str]], predecessor_map: dict[str, set[str]]) -> tuple[dict[str, Optional[str]], dict[str, set[str]]]:
    # The immediate dominator of a basic block b
    # is the dominator d_i in dominators(b) s.t.
    # dominators(d_i) intersection dominators(b) is maximal
    # Note that the immediate dominator of a basic block b
    # must be unique since otherwise if dominators(d_i) intersection dominators(b)
    # has the same length but is distinct from dominators(d_j) intersection dominators(b)
    # it means that there exists a dominator of b that does not dominate both d_i and d_j
    # and that is a contradiction since all dominators of b must dominate all predecessors
    # of b.
    bb_to_imdom_map : dict[str, Optional[str]] = {}
    imdom_to_bbs_map: dict[str, set[str]] = defaultdict(set)
    entry_node: Optional[str] = None
    for bb, dominators in dominator_map.items():
        if len(predecessor_map[bb]) == 0:
            # Entry block
            entry_node = bb
            bb_to_imdom_map[bb] = None
        elif len(predecessor_map[bb]) == 1:
            # Must be the predecessor
            predecessor_bb = next(iter(predecessor_map[bb]))
            assert predecessor_bb in dominators, f"{bb=} has predecessor map {predecessor_map[bb]=} but {predecessor_bb=} not in dominators {dominators=}"
            bb_to_imdom_map[bb] = predecessor_bb
            imdom_to_bbs_map[predecessor_bb].add(bb)
        else:
            # Now find a dominator d_i in dominators(b) s.t. it is dominated by all the other
            # dominators(b)
            max_dominators = 0
            current_imdom = None
            for candidate_imdom in dominators:
                if (candidate_max_dominators := len(dominator_map[candidate_imdom] & dominators)) and candidate_max_dominators > max_dominators:
                    max_dominators = candidate_max_dominators
                    current_imdom = candidate_imdom
            assert current_imdom is not None, f"Expected to find an immediate dominator for {bb=}, {dominator_map=}"
            bb_to_imdom_map[bb] = current_imdom
            imdom_to_bbs_map[current_imdom].add(bb)
    assert entry_node is not None, f"Entry node must dominate all basic blocks, but could not find it. {dominator_map=}"
    assert next(iter(bb_to_imdom_map.keys())) == entry_node, f"The entry node {entry_node=} must be the first key in the dominator map. Got {dominator_map=}"
    return bb_to_imdom_map, imdom_to_bbs_map

# Intraprocedural DCE, global
def global_dead_code_elimination():
    pass

def main():
    program = json.load(sys.stdin)
    for fn in program["functions"]:
        fn_cfg = fn["cfg"]
        breakpoint()
        fn_bb_predecessor_map = get_predecessor_map(fn_cfg)
        fn_cfg_dominators = get_dominators(fn_cfg, fn_bb_predecessor_map)
        bb_to_imdom_map, imdom_to_bbs_map = get_imdom_and_dominator_tree(fn_cfg_dominators, fn_bb_predecessor_map)
        breakpoint()


if __name__ == "__main__":
    main()
