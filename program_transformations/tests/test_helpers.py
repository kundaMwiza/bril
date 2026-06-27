from __future__ import annotations

from typing import Any


def compare_dicts(map_one: dict[str, Any], map_two: dict[str, Any]):
    def check_and_dispatch(iter_one, iter_two):
        assert type(iter_one) == type(iter_two), (
            f"{type(iter_one)=} is not equal to {type(iter_two)=}"
        )
        if isinstance(iter_one, dict):
            compare_dicts(iter_one, iter_two)
        elif isinstance(iter_one, (list, tuple, set)):
            for i, (next_iter_one, next_iter_two) in enumerate(zip(iter_one, iter_two)):
                try:
                    check_and_dispatch(next_iter_one, next_iter_two)
                except AssertionError as e:
                    raise AssertionError(
                        f"Index {i=} mismatch for {k_one=}, {next_iter_one=}, {next_iter_two=}"
                    ) from e
        else:
            assert iter_one == iter_two, (
                f"{iter_one == iter_two=} mismatch, {iter_one=}, {iter_two=}"
            )

    assert type(map_one) == type(map_two)
    for k_one, v_one in map_one.items():
        v_two = map_two[k_one]
        assert type(v_one) == type(v_two)
        assert k_one in map_two, f"Did not find {k_one=} in {map_two.keys()=}"
        check_and_dispatch(v_one, v_two)
