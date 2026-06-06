try:
    from . import cfg, dataflow_analysis
    from .cfg import CFG, InstructionBase, Program
except ImportError:
    import cfg
    import dataflow_analysis
    from cfg import CFG, InstructionBase, Program


def insert_ssa_instructions(cfg: CFG, dataflow_information):
    pass


def convert_to_ssa(cfg: CFG) -> CFG:
    dataflow_information, undef_map = dataflow_analysis.run_dataflow_analysis(cfg)
    if undef_map:
        cfg.get_root_block().insert_instrs(
            [
                InstructionBase.create_undef_instr(arg_name, arg_dtype)
                for arg_name, arg_dtype in undef_map.items()
            ]
        )
        dataflow_information, undef_map = dataflow_analysis.run_dataflow_analysis(cfg)
        assert not undef_map
    cfg.get_dominator_info()
    return cfg
