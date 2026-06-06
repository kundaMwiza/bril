from typing import TypeAlias, Any, Generator


def print_digraph(name:str, cfg) -> None:
    defined_nodes = set()
    def_lines : list[str] = []
    edge_lines : list[str] = []

    new_cfg = {}
    for k, v in cfg.items():
        new_k = k.replace(".", "_")
        new_v = [_v.replace(".", "_") for _v in v]
        new_cfg[new_k] = new_v

    for label, successors in new_cfg.items():
        if label not in defined_nodes:
            def_lines.append(f"\t {label};")
        for s in successors:
            edge_lines.append(f"\t {label} -> {s};")

    print(f"digraph {name} {{")
    if def_lines:
        print("\n".join(def_lines))
    if edge_lines:
        print("\n".join(edge_lines))
    print(f"}}")
