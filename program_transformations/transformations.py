from __future__ import annotations

import copy
import dataclasses
import json
from typing import Any, Callable

try:
    from . import cfg, dataflow_analysis
    from .cfg import Instruction, Program
    from .ssa import convert_to_ssa
except ImportError:
    import cfg
    import dataflow_analysis
    from cfg import CFG, Program
    from ssa import convert_to_ssa

AVAILABLE_TRANSFORMATIONS: dict[str, Transformation] = {}


def register_transformation(transformation: type[Transformation]):
    if transformation.__name__ in AVAILABLE_TRANSFORMATIONS:
        raise RuntimeError(f"{transformation.__name__} already registered")
    AVAILABLE_TRANSFORMATIONS[transformation.__name__] = transformation()
    return transformation


@dataclasses.dataclass
class TransformationStage:
    name: str
    program: Program | dict[Any, Any]


class Transformation:
    def __call__(self, stage: TransformationStage) -> TransformationStage:
        stage_copy = copy.deepcopy(stage)
        new_program = self.apply(stage_copy.program)
        return TransformationStage(type(self).__name__, new_program)

    def apply(self, program: Program | dict[Any, Any]) -> Any:
        raise NotImplementedError


class BrilJsonToProgram(Transformation):
    def apply(self, program) -> Program:
        assert isinstance(program, dict)
        return cfg.Program.from_bril_dict(program)


class ProgramToBrilJson(Transformation):
    def apply(self, program) -> dict[str, Any]:
        assert isinstance(program, Program)
        result = program.to_bril_dict()
        return result


@register_transformation
class ProgramToSSA(Transformation):
    def apply(self, program) -> Program:
        assert isinstance(program, Program)
        new_cfgs: list[CFG] = []
        breakpoint()
        for cfg in program.cfgs:
            new_cfgs.append(convert_to_ssa(cfg))
        program.cfgs = new_cfgs
        return program


TRANSFORMATION_PRESETS: dict[str, list[type[Transformation]]] = {
    "default": [ProgramToSSA]
}


def apply_transformations(
    bril_input_program: dict[str, Any],
    transformations: list[Transformation],
    preset: str | None = None,
    print_output: bool = True,
) -> list[Program]:
    assert not (transformations and preset), (
        "Only only one of `preset` and `transformation` must be provided"
    )
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
