from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go


def save_eigenspectrum_html(eigenvalues, output_path: str | Path, title: str) -> None:
    output_path = Path(output_path)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=list(range(len(eigenvalues))),
            y=eigenvalues,
            mode='lines+markers',
            name='eigenvalues',
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title='component index',
        yaxis_title='eigenvalue',
        template='plotly_white',
    )
    figure.write_html(str(output_path), include_plotlyjs='cdn')
