# Guide 5: Topology Visualization & Graph Export

Understanding the architectural innovations discovered by neuroevolution requires clear, publication-quality visualizations.

Malthus-NEAT provides two complementary visualization interfaces:
1. **Interactive Matplotlib Subplot Rendering (`plot_network`)**: Seamlessly embeds the network into multi-panel diagnostic figures.
2. **Graphviz DOT Export (`visualize`)**: Generates vector SVG, PNG, or PDF graph diagrams with automatic hierarchical layout.

---

## 1. Visualizing with Matplotlib: `plot_network()`

The `plot_network()` method extracts the active subgraph of the best-evolved network and renders it directly onto any Matplotlib `Axes`:

```python
import matplotlib.pyplot as plt
import numpy as np
from malthus_neat.learn import NEATClassifier

# Fit classifier
clf = NEATClassifier(pop_size=50, max_generations=60, random_state=42)
X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
y = np.array([0, 1, 1, 0], dtype=np.int32)
clf.fit(X, y)

# Render side-by-side: Decision Boundary + Topology
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 0: Decision Boundary
# (draw contour on axes[0]...)

# Panel 1: Evolved Topology
clf.plot_network(ax=axes[1])
axes[1].set_title("Evolved Minimal Architecture")

plt.tight_layout()
plt.savefig("network_diagnostic.png", dpi=200)
```

### Visual Encoding Conventions
In `plot_network`:
- **Green Nodes**: Input features ($x_0, x_1, \dots$).
- **Yellow Nodes**: Learnable bias unit ($b$).
- **Blue Nodes**: Hidden neurons discovered via edge splits.
- **Red Nodes**: Output units ($\hat{y}$).
- **Solid Blue Edges**: Excitatory synaptic weights ($w > 0$).
- **Dashed Red Edges**: Inhibitory synaptic weights ($w < 0$).
- **Line Width**: Scaled proportionally to synaptic weight magnitude $|w|$.

---

## 2. Vector SVG & Graphviz Export: `visualize()`

For publications and high-resolution reports, `visualize()` renders graphs using Graphviz DOT layouts:

```python
# Export vector SVG diagram
clf.visualize(filepath="architecture.svg", format="svg")

# Export publication-quality PDF
clf.visualize(filepath="architecture.pdf", format="pdf")
```

If Graphviz is not installed locally, `visualize()` automatically falls back to NetworkX Matplotlib rendering to ensure zero failed pipelines.

---

## 3. Extracting NetworkX DiGraph Directly

For custom algorithmic graph analysis (e.g. computing graph diameter, betweenness centrality, or clustering coefficients), extract the `NetworkX` directed graph directly:

```python
from malthus_neat.interpreters.visualizer import networkx_from_genome

# Convert best genome to NetworkX DiGraph
G = networkx_from_genome(clf.best_genome_, clf.config_)

print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# Inspect edge attributes
for u, v, data in G.edges(data=True):
    print(f"Synapse {u} -> {v}: weight = {data['weight']:.3f}")
```
