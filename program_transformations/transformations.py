from __future__ import annotations

from typing import Callable, Any
import json
import copy
import dataclasses

try:
    from .cfg import Program, InstructionBase
    from . import cfg
    from . import dataflow_analysis
except ImportError:
    from cfg import Program, CFG
    import cfg
    import dataflow_analysis

AVAILABLE_TRANSFORMATIONS : dict[str, Transformation] = {}

def register_transformation(transformation: type[Transformation]):
    if transformation.__name__ in AVAILABLE_TRANSFORMATIONS:
        raise RuntimeError(f"{transformation.__name__} already registered")
    AVAILABLE_TRANSFORMATIONS[transformation.__name__] = transformation()
    return transformation

@dataclasses.dataclass
class TransformationStage:
    name: str
    program: Program

class Transformation:
    def __call__(self, stage: TransformationStage) -> TransformationStage: 
        stage_copy = copy.deepcopy(stage)
        new_program = self.apply(stage_copy.program)
        return TransformationStage(type(self).__name__, new_program)

    def apply(self, stage: TransformationStage) -> TransformationStage | None:
        raise NotImplementedError

class BrilJsonToProgram(Transformation):
    def apply(self, program:  dict[str, Any]) -> Program:
        assert isinstance(program, dict)
        return cfg.Program.from_bril_dict(program)
    
class ProgramToBrilJson(Transformation):
    def apply(self, program: Program) -> None:
        assert isinstance(program, Program)
        result = program.to_bril_dict()
        return result
    
@register_transformation
class ProgramToSSA(Transformation):
    def apply(self, program: Program) -> None:
        assert isinstance(program, Program)
        new_cfgs: list[CFG] = []
        for cfg in program.cfgs:
            dataflow_information, undef_map = dataflow_analysis.run_dataflow_analysis(cfg)
            if undef_map:
                cfg.get_root_block().insert_instrs(
                    [InstructionBase.create_undef_instr(arg_name, arg_dtype) for arg_name, arg_dtype in undef_map.items()]
                )
                dataflow_information, undef_map = dataflow_analysis.run_dataflow_analysis(cfg)
                assert not undef_map 
            dominance_info = cfg.get_dominator_info()
        pass
   
        
TRANSFORMATION_PRESETS : dict[str, list[Transformation]] = {
    "default": [ProgramToSSA]
}

def apply_transformations(bril_input_program: dict[str, Any], transformations: list[Transformation], preset: str | None = None,  print_output: bool = True) -> list[Program]:
    assert not (transformations and preset), "Only only one of `preset` and `transformation` must be provided"
    if preset is not None:
        assert preset in TRANSFORMATION_PRESETS
        transformations = TRANSFORMATION_PRESETS[preset]
        
    # Convert to internal IR
    new_stage = BrilJsonToProgram()(TransformationStage("original", bril_input_program))
    intermediate_programs = [new_stage]
    breakpoint()
    for t in transformations:
        t_cls = AVAILABLE_TRANSFORMATIONS[t.__name__]
        _next_stage = t_cls(new_stage)
        intermediate_programs.append(_next_stage)
        new_stage = _next_stage

    # Output to json str -> can piped into `brili`
    last_stage = ProgramToBrilJson()(new_stage)
    intermediate_programs.append(last_stage)
    print(json.dumps(last_stage.program, indent=2))
    return intermediate_programs 




    
        
    
