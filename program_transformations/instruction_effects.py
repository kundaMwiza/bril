from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AbstractHeap:
    name: str
    children: list[AbstractHeap] = field(default_factory=list)
    # dfs pre-order and post-order index
    pre: int = -1
    post: int = -1

    def add_children(
        self, children: AbstractHeap | list[AbstractHeap]
    ) -> AbstractHeap | list[AbstractHeap]:
        if isinstance(children, AbstractHeap):
            self.children.append(children)
        else:
            assert isinstance(children, list)
            self.children.extend(children)
        return children

    @property
    def interval(self) -> tuple[int, int]:
        if self.pre < 0 or self.post < 0:
            raise RuntimeError("The abstract heap has not been numbered yet")
        return (self.pre, self.post)

    def number_dfs(self, start: int = 0) -> int:
        # This function must return a dfs pre-order index that has not yet
        # been allocated
        i = start
        self.pre = i
        i += 1
        for c in self.children:
            i = c.number_dfs(i)
        self.post = i
        i += 1
        return i

    def __repr__(self):
        return f"AbstractHeap({self.name}, pre={self.pre}, post={self.post}, num_children={len(self.children)})"


World = AbstractHeap("World")
Memory = AbstractHeap("Memory")
SSAState = AbstractHeap("SSAState")
Control = AbstractHeap("Control")
IO = AbstractHeap("IO")
Stack = AbstractHeap("Stack")
Heap = AbstractHeap("Heap")
Memory.add_children([Stack, Heap])
World.add_children([Memory, SSAState, Control, IO])
World.number_dfs()


@dataclass
class Effects:
    reads: list[AbstractHeap] = field(default_factory=list)
    writes: list[AbstractHeap] = field(default_factory=list)

    def add_write(self, ah: AbstractHeap):
        self.writes.append(ah)

    def add_read(self, ah: AbstractHeap):
        self.reads.append(ah)

    def reads_from(self, ah: AbstractHeap):
        return ah in self.reads

    def writes_to(self, ah: AbstractHeap):
        return ah in self.writes
