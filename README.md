# Volatility Surface Visualizer

A Python tool to visualize implied volatility surfaces for options trading, supporting both Alpaca and Polygon.io data sources.

## Features
- **3D Visualization**: Interactive 3D plots of volatility surfaces.
- **Multiple Data Sources**: Support for Alpaca and Polygon.io APIs.
- **Streamlit App**: Web-based interface for easy interaction.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Streamlit App
Run the interactive dashboard:
```bash
streamlit run streamlit_app.py
```

### Scripts
- `alpaca_volatility_surface.py`: Fetch and plot using Alpaca data.
- `polygon_volatility_surface.py`: Fetch and plot using Polygon.io data.
- `live_volatility_surface.py`: Real-time volatility updates.

## License
MIT License
