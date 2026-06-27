from __future__ import annotations

import dataclasses
import enum
from typing import cast

try:
    from .cfg import Dtype, Instruction, PlaceholderInstr
except ImportError:
    from cfg import Dtype, Instruction, PlaceholderInstr


class OpArgTypeResolveFailure(enum.Enum):
    NotRequired = enum.auto()
    RequiresDefinition = enum.auto()


ARG_TYPE_RESOLVERS: dict[str, ResolveArgType] = {}


def register_arg_type_resolver(instr_name: str | list[str] | tuple[str]):
    if isinstance(instr_name, str):
        instr_name = [instr_name]

    def _register(cls: type):
        for i in instr_name:
            ARG_TYPE_RESOLVERS[instr_name] = cls
        return cls

    return _register


class ResolveArgType:
    @classmethod
    def _impl(cls, arg_name: str, instr: Instruction):
        raise NotImplementedError

    @classmethod
    def resolve(cls, arg_name: str, instr: Instruction):
        assert not isinstance(instr, PlaceholderInstr)
        cls._impl(arg_name, instr)


@register_arg_type_resolver(("add", "mul", "sub", "div"))
class ArithmeticOpResolver(ResolveArgType):
    @classmethod
    def _impl(cls, arg_name, instr):
        return int


@register_arg_type_resolver(("eq", "lt", "gt", "le", "ge"))
class ComparisonOpResolver(ResolveArgType):
    @classmethod
    def _impl(self, arg_name, instr):
        return int


@register_arg_type_resolver(("not", "and", "or"))
class LogicOpResolver(ResolveArgType):
    @classmethod
    def _impl(self, arg_name, instr):
        return bool


@register_arg_type_resolver(("jmp", "br", "call", "ret"))
class ControlOpResolver(ResolveArgType):
    @classmethod
    def _impl(self, arg_name: str, instr: Instruction):
        assert isinstance(instr, Instruction)
        instr = cast(Instruction, instr)
        if instr.op == "jmp":
            return OpArgTypeResolveFailure.NotRequired
        elif instr.op == "br":
            return bool
        elif instr.op == "call":
            assert len(instr.funcs) == 1
            fn = instr.funcs[0]
            entire_program = instr.basic_block.cfg.program
            for cfg in entire_program.cfgs:
                if cfg.name == fn:
                    # Well its invalid to have more than one parameter
                    # with the same name so hope for the best here
                    for arg_instr in cfg.get_cfg_arguments():
                        if arg_instr.dest == arg_name:
                            return arg_instr.dtype
                    raise NotImplementedError


@register_arg_type_resolver(("id", "print", "nop"))
class MiscOpResolver(ResolveArgType):
    @classmethod
    def _impl(self, arg_name: str, instr: Instruction):
        assert isinstance(instr, Instruction)
        instr = cast(Instruction, instr)
        if instr.op == "id":
            return instr.dtype
        elif instr.op == "print":
            return OpArgTypeResolveFailure.RequiresDefinition
        elif instr.op == "nop":
            return OpArgTypeResolveFailure.NotRequired


def maybe_resolve_arg_type(
    arg_name: str, instr: Instruction
) -> Dtype | OpArgTypeResolveFailure:
    if instr.op in ARG_TYPE_RESOLVERS:
        return ARG_TYPE_RESOLVERS[instr.op].resolve(arg_name, instr)
    # TODO support Floating point, Memory, Character, Bit casting, Dynamic e.t.c
    raise NotImplementedError("Other operation classes are not yet implemented")
