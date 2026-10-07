"""Network topology extraction and visualization for NeatGenome."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from malthus_neat.genomes.neat_genome import NeatGenome, NeatGenomeConfig
from malthusjax.core.logger import get_logger

logger = get_logger("malthus_neat.interpreters.visualizer")


def extract_network_graph(
    genome: NeatGenome, config: NeatGenomeConfig
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extracts node metadata and active connections from a NeatGenome.

    Returns:
        nodes: List of dicts with 'id', 'type', 'layer'
        edges: List of dicts with 'in', 'out', 'weight'
    """
    nodes_arr = np.array(genome.nodes)
    conns_arr = np.array(genome.conns)

    # 1. Parse active nodes
    nodes: List[Dict[str, Any]] = []
    n_inputs = config.num_inputs
    n_outputs = config.num_outputs

    for i in range(nodes_arr.shape[0]):
        if not np.isnan(nodes_arr[i, 0]):
            node_id = i
            if node_id < n_inputs:
                node_type = "input"
                layer = 0.0
            elif node_id < n_inputs + n_outputs:
                node_type = "output"
                layer = 1.0
            else:
                node_type = "hidden"
                layer = 0.5
            nodes.append({"id": node_id, "type": node_type, "layer": layer})

    # 2. Parse active connections
    edges: List[Dict[str, Any]] = []
    for i in range(conns_arr.shape[0]):
        if not np.isnan(conns_arr[i, 0]):
            # In TensorNEAT DefaultConn:
            # fixed_attrs: ['input_index', 'output_index'], custom_attrs: ['weight']
            # Depending on column layout, extract in_node, out_node, weight
            weight = float(conns_arr[i, 0])
            in_node = int(conns_arr[i, 1]) if conns_arr.shape[1] > 1 else 0
            out_node = int(conns_arr[i, 2]) if conns_arr.shape[1] > 2 else 0
            edges.append({"in": in_node, "out": out_node, "weight": weight})

    return nodes, edges


def to_networkx(genome: NeatGenome, config: NeatGenomeConfig) -> Any:
    """Converts a NeatGenome into a NetworkX DiGraph."""
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "networkx is required for to_networkx(). Install via pip install networkx."
        )

    G = nx.DiGraph()
    nodes, edges = extract_network_graph(genome, config)

    for n in nodes:
        G.add_node(n["id"], node_type=n["type"], layer=n["layer"])

    for e in edges:
        G.add_edge(e["in"], e["out"], weight=e["weight"])

    return G


def to_dot(genome: NeatGenome, config: NeatGenomeConfig) -> str:
    """Generates a Graphviz DOT representation of the neural network topology."""
    nodes, edges = extract_network_graph(genome, config)
    lines = ["digraph NEAT {", "  rankdir=LR;", "  node [shape=circle];"]

    # Group by layer
    inputs = [n["id"] for n in nodes if n["type"] == "input"]
    outputs = [n["id"] for n in nodes if n["type"] == "output"]

    if inputs:
        lines.append(f"  subgraph {{ rank=source; {' '.join(str(i) for i in inputs)}; }}")
    if outputs:
        lines.append(f"  subgraph {{ rank=sink; {' '.join(str(o) for o in outputs)}; }}")

    for n in nodes:
        color = (
            "lightblue"
            if n["type"] == "input"
            else ("lightcoral" if n["type"] == "output" else "lightgreen")
        )
        lines.append(f'  {n["id"]} [label="Node {n["id"]}", style=filled, fillcolor={color}];')

    for e in edges:
        style = "solid" if e["weight"] >= 0 else "dashed"
        penwidth = min(max(abs(e["weight"]) * 1.5, 0.5), 5.0)
        color = "blue" if e["weight"] >= 0 else "red"
        lines.append(
            f'  {e["in"]} -> {e["out"]} [label="{e["weight"]:.2f}", style={style}, penwidth={penwidth:.1f}, color={color}];'
        )

    lines.append("}")
    return "\n".join(lines)


def plot_network(
    genome: NeatGenome,
    config: NeatGenomeConfig,
    path: Optional[str] = None,
    view: bool = False,
    ax: Optional[Any] = None,
) -> Any:
    """Renders and optionally saves the network topology plot using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plot_network().")

    nodes, edges = extract_network_graph(genome, config)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        created_fig = True
    else:
        fig = ax.get_figure()

    # Compute positions (X: layer, Y: vertical spread)
    inputs = [n for n in nodes if n["type"] == "input"]
    hidden = [n for n in nodes if n["type"] == "hidden"]
    outputs = [n for n in nodes if n["type"] == "output"]

    pos: Dict[int, Tuple[float, float]] = {}

    def layout_layer(node_list: List[Dict[str, Any]], x_val: float) -> None:
        k = len(node_list)
        if k == 0:
            return
        y_vals = np.linspace(-1.0, 1.0, k) if k > 1 else [0.0]
        for n, y in zip(node_list, y_vals):
            pos[n["id"]] = (x_val, y)

    layout_layer(inputs, 0.0)
    layout_layer(hidden, 0.5)
    layout_layer(outputs, 1.0)

    # Draw edges
    for e in edges:
        if e["in"] in pos and e["out"] in pos:
            p1 = pos[e["in"]]
            p2 = pos[e["out"]]
            color = "blue" if e["weight"] >= 0 else "red"
            alpha = min(max(abs(e["weight"]) / 2.0, 0.2), 1.0)
            ax.annotate(
                "",
                xy=p2,
                xytext=p1,
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    alpha=alpha,
                    lw=min(max(abs(e["weight"]), 0.5), 3.0),
                ),
            )

    # Draw nodes
    for n in nodes:
        if n["id"] in pos:
            x, y = pos[n["id"]]
            c = (
                "skyblue"
                if n["type"] == "input"
                else ("salmon" if n["type"] == "output" else "lightgreen")
            )
            ax.scatter(x, y, s=400, color=c, edgecolors="black", zorder=3)
            ax.text(
                x,
                y,
                str(n["id"]),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                zorder=4,
            )

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-1.3, 1.3)
    ax.axis("off")
    ax.set_title(
        f"NEAT Topology: {len(nodes)} Nodes ({len(inputs)} In, {len(hidden)} Hidden, {len(outputs)} Out), "
        f"{len(edges)} Synapses",
        fontsize=12,
    )

    if created_fig:
        plt.tight_layout()
        if path:
            plt.savefig(path, dpi=200)
            logger.info("Saved topology plot to %s", path)
        if not view:
            plt.close(fig)
    return ax


def draw_network(
    genome: NeatGenome,
    config: Optional[NeatGenomeConfig] = None,
    filepath: Optional[str] = None,
    format: str = "svg",
    feature_names: Optional[Any] = None,
    view: bool = False,
) -> Any:
    """Renders or saves the NEAT network topology.

    Args:
        genome: NeatGenome individual.
        config: Optional NeatGenomeConfig instance.
        filepath: Optional path to save the generated figure.
        format: File format ("svg", "png", "pdf").
        feature_names: Optional human-readable feature labels.
        view: Whether to display interactively.

    Returns:
        The output filepath or None.
    """
    if config is None:
        config = NeatGenomeConfig()
    if filepath is not None and "." not in str(filepath):
        filepath = f"{filepath}.{format}"
    plot_network(genome, config, path=filepath, view=view)
    return filepath


__all__ = ["extract_network_graph", "to_networkx", "to_dot", "plot_network", "draw_network"]
