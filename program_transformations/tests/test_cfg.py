import pytest
from pathlib import Path
import json
from .. import transformations
from . import test_helpers

EIGHT_QUEENS = Path(__file__).resolve().parents[2] / "test" / "print" / "eight-queens.json"


@pytest.mark.parametrize(
    "program", [EIGHT_QUEENS]
)
def test_roundtrip(program: Path):
    assert program.exists()
    original_program = json.loads(program.read_text())
    intermediate_programs: transformations.TransformationStage = transformations.apply_transformations(original_program, [])
    last_stage = intermediate_programs[-1]
    assert last_stage.name == transformations.ProgramToBrilJson.__name__
    test_helpers.compare_dicts(original_program, last_stage.program)
