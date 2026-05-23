"""
Wizualizacje Plotly dla modelu MLP.
Skopiowane z LAB_5_WSI/Code/Visualization/network_visualization.py i zaadaptowane.
"""

import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MAX_VISIBLE_NODES = 8

LAYER_COLORS = {
    'input':   '#636EFA',
    'relu':    '#00CC96',
    'sigmoid': '#AB63FA',
    'linear':  '#FFA15A',
}

_DARK = '#1e1e2e'


# ── Architektura 2D ────────────────────────────────────────────────────────────

def pokaz_architekture(layer_sizes, activations):
    act_labels  = ['input'] + list(activations)
    num_layers  = len(layer_sizes)
    x_positions = [i / (num_layers - 1) for i in range(num_layers)]

    node_x, node_y, node_text, node_color = [], [], [], []
    edge_x,  edge_y  = [], []
    layer_node_positions = []

    for col, (size, act) in enumerate(zip(layer_sizes, act_labels)):
        visible = min(size, MAX_VISIBLE_NODES)
        y_vals  = ([v / (visible - 1) for v in range(visible)] if visible > 1 else [0.5])
        y_vals  = [y - 0.5 for y in y_vals]
        color   = LAYER_COLORS.get(act.lower(), '#EF553B')
        positions = []
        for y in y_vals:
            node_x.append(x_positions[col])
            node_y.append(y)
            node_text.append(f"Layer {col} ({act})<br>Size: {size}"
                             + (" [truncated]" if size > MAX_VISIBLE_NODES else ""))
            node_color.append(color)
            positions.append((x_positions[col], y))
        layer_node_positions.append(positions)

    for i in range(len(layer_node_positions) - 1):
        for x1, y1 in layer_node_positions[i]:
            for x2, y2 in layer_node_positions[i + 1]:
                edge_x += [x1, x2, None]
                edge_y += [y1, y2, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                             line=dict(color='rgba(150,150,150,0.25)', width=0.8),
                             hoverinfo='none', name='connections'))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers',
                             marker=dict(size=14, color=node_color,
                                         line=dict(width=1, color='white')),
                             text=node_text,
                             hovertemplate='%{text}<extra></extra>', name='nodes'))
    for col, (size, act) in enumerate(zip(layer_sizes, act_labels)):
        fig.add_annotation(x=x_positions[col], y=0.62,
                           text=f"<b>{act.capitalize()}</b><br>{size} nodes",
                           showarrow=False, font=dict(size=11), xref='x', yref='y')
    fig.update_layout(title='Architektura sieci neuronowej', showlegend=False,
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      plot_bgcolor=_DARK, paper_bgcolor=_DARK,
                      font=dict(color='white'), margin=dict(l=20, r=20, t=60, b=20))
    fig.show()


# ── Architektura 3D ────────────────────────────────────────────────────────────

def pokaz_architekture_3d(layer_sizes, activations):
    act_labels  = ['input'] + list(activations)
    num_layers  = len(layer_sizes)

    x_gap        = 3.0
    ring_radius  = 1.2          # base radius — scales with sqrt(size/16)
    MAX_EDGES_PER_PAIR = 600

    node_x, node_y, node_z           = [], [], []
    node_colors, node_sizes, node_labels = [], [], []
    edge_x, edge_y, edge_z           = [], [], []
    layer_positions                  = []

    for col, (size, act) in enumerate(zip(layer_sizes, act_labels)):
        color    = LAYER_COLORS.get(act.lower(), '#EF553B')
        x_pos    = col * x_gap
        radius   = ring_radius * max(1.0, math.sqrt(size / 16))
        dot_size = max(3, int(10 - math.log2(max(size, 1))))

        positions = []
        for k in range(size):
            if size == 1:
                y, z = 0.0, 0.0
            else:
                angle = 2 * math.pi * k / size
                y = radius * math.cos(angle)
                z = radius * math.sin(angle)
            node_x.append(x_pos)
            node_y.append(y)
            node_z.append(z)
            node_colors.append(color)
            node_sizes.append(dot_size)
            node_labels.append(
                f"<b>{act.capitalize()} layer</b><br>"
                f"Warstwa {col} | węzeł {k + 1}/{size}"
            )
            positions.append((x_pos, y, z))
        layer_positions.append(positions)

    # Krawędzie — próbkowanie żeby przeglądarka nie zwalniała
    import random as _rng
    _rng.seed(42)
    for i in range(len(layer_positions) - 1):
        pairs = [(l, r) for l in layer_positions[i] for r in layer_positions[i + 1]]
        if len(pairs) > MAX_EDGES_PER_PAIR:
            pairs = _rng.sample(pairs, MAX_EDGES_PER_PAIR)
        for (x1, y1, z1), (x2, y2, z2) in pairs:
            edge_x += [x1, x2, None]
            edge_y += [y1, y2, None]
            edge_z += [z1, z2, None]

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, mode='lines',
        line=dict(color='rgba(180,180,180,0.12)', width=0.8),
        hoverinfo='none', name='connections'))

    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z, mode='markers',
        marker=dict(size=node_sizes, color=node_colors,
                    line=dict(width=0), opacity=0.90),
        text=node_labels,
        hovertemplate='%{text}<extra></extra>', name='nodes'))

    # Etykiety warstw
    lx, ly, lz, lt = [], [], [], []
    for col, (size, act) in enumerate(zip(layer_sizes, act_labels)):
        r = ring_radius * max(1.0, math.sqrt(size / 16))
        lx.append(col * x_gap)
        ly.append(r + 0.6)
        lz.append(0.0)
        lt.append(f"{act.capitalize()}<br>{size} węzłów")
    fig.add_trace(go.Scatter3d(
        x=lx, y=ly, z=lz, mode='text', text=lt,
        textfont=dict(size=11, color='white'),
        hoverinfo='none', name='labels'))

    _ax = dict(showgrid=False, zeroline=False, showticklabels=False,
               showbackground=True, backgroundcolor=_DARK, gridcolor='#333')

    fig.update_layout(
        width=50,   # <--- TUTAJ DODANE (Zwężenie szerokości do 750px)
        height=50,  # <--- TUTAJ DODANE (Usztywnienie wysokości, żeby kula nie stała się jajem)
        title=dict(text='Architektura sieci (3D)', font=dict(size=20, color='white')),
        scene=dict(
            xaxis=dict(**_ax, title='Warstwy'),
            yaxis=dict(**_ax, title=''),
            zaxis=dict(**_ax, title=''),
            bgcolor=_DARK,
            camera=dict(eye=dict(x=1.6, y=-1.8, z=1.0), up=dict(x=0, y=0, z=1)),
        ),
        paper_bgcolor=_DARK, font=dict(color='white'),
        showlegend=False, margin=dict(l=0, r=0, t=50, b=0))
    fig.show()


# ── Postęp trenowania ─────────────────────────────────────────────────────────

def pokaz_postep_trenowania(loss_history, val_mse_history):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Training MSE', 'Validation MSE'))
    xs = list(range(1, len(loss_history) + 1))
    fig.add_trace(go.Scatter(x=xs, y=loss_history, name='Train MSE',
                             line=dict(color='#636EFA')), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=val_mse_history, name='Val MSE',
                             line=dict(color='#EF553B')), row=1, col=2)
    fig.update_xaxes(title_text='Checkpoint', gridcolor='#333')
    fig.update_yaxes(gridcolor='#333', type='log', tickformat='.2e')

    fig.update_layout(
        width=50, # <--- TUTAJ DODANE (Zwężenie szerokości do 750px)
        title='Postęp trenowania (skala log)',
        plot_bgcolor=_DARK, paper_bgcolor=_DARK,
        font=dict(color='white'), height=400)
    fig.show()
