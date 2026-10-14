# Colorbar Configuration Reference

## Overview
The PyVista postprocessing script uses a standardized colorbar configuration with consistent **viridis** colormap across all visualizations for professional, perceptually-uniform color representation.

## Configuration Specifications

### Colormap
- **Standard Colormap**: viridis
  - Perceptually uniform
  - Color-blind friendly
  - Print-friendly (converts to grayscale well)
  - Applied to ALL visualizations for consistency

### Font Settings
- **Font Family**: Arial
- **Font Size**: 12pt (both title and labels)
- **Font Style**: Bold
- **Font Color**: Black

### Colorbar Layout
- **Orientation**: Vertical
- **Title Position**: Right side of colorbar (vertical orientation)
- **Position**: Right side, vertically centered
  - `position_x`: 0.88 (88% from left edge)
  - `position_y`: 0.25 (25% from bottom edge)
- **Dimensions**:
  - Width: 8% of plot width
  - Height: 50% of plot height

### Tick and Label Formatting
- **Number of Labels**: 5 evenly spaced ticks
- **Number Format**: Scientific notation (`%.2e`)
  - Example: `1.23e-05` instead of `0.0000123`
  - 2 decimal places for precision

### Implementation
The configuration is centralized in the `_get_scalar_bar_args()` method:

```python
def _get_scalar_bar_args(self, title):
    """Get standardized scalar bar configuration."""
    return {
        "title": title,
        "title_font_size": 12,
        "label_font_size": 12,
        "font_family": "arial",
        "bold": True,
        "italic": False,
        "color": "black",
        "n_labels": 5,
        "fmt": "%.2e",
        "vertical": True,
        "position_x": 0.88,
        "position_y": 0.25,
        "width": 0.08,
        "height": 0.5,
        "unconstrained_font_size": True,
    }
```

## Applied to All Visualizations

All plots now use the **viridis** colormap for consistency:

1. **Displacement Magnitude**
   - Title: "Displacement (m)"
   - Colormap: viridis

2. **Von Mises Stress**
   - Title: "Von Mises Stress (Pa)"
   - Colormap: viridis

3. **Stress Components**
   - Titles: "Stress XX (Pa)", "Stress YY (Pa)", "Stress XY (Pa)"
   - Colormap: viridis

4. **Strain Components**
   - Titles: "Strain XX", "Strain YY", "Strain XY"
   - Colormap: viridis

## Why Viridis?

The viridis colormap was chosen for:
- **Perceptual uniformity**: Equal changes in data appear as equal changes in color
- **Accessibility**: Works well for color-blind viewers
- **Print quality**: Converts to grayscale gracefully
- **Scientific standard**: Widely used in scientific publications
- **Monotonic luminance**: Preserves information even in grayscale

## Customization
To modify the colorbar appearance, edit the `_get_scalar_bar_args()` method in `postprocess_pyvista.py`.

### Common Adjustments

#### Change Colormap
To change from viridis to another colormap, modify the `cmap` parameter in each plotting method:
```python
cmap="plasma",    # Purple/orange
cmap="inferno",   # Dark red/yellow
cmap="cividis",   # Alternative color-blind friendly
cmap="coolwarm",  # Diverging blue-red
```

#### Change Font Size
```python
"title_font_size": 14,  # Larger title
"label_font_size": 10,  # Smaller labels
```

#### Reposition Colorbar
```python
"position_x": 0.85,  # Move slightly left
"position_y": 0.30,  # Move up
```

#### Change Number Format
```python
"fmt": "%.3f",   # Decimal notation with 3 decimals
"fmt": "%.1e",   # Scientific notation with 1 decimal
```

#### Adjust Number of Ticks
```python
"n_labels": 7,  # More tick labels
```

## PyVista scalar_bar_args Reference
For complete documentation, see: https://docs.pyvista.org/version/stable/api/plotting/_autosummary/pyvista.Plotter.add_mesh.html
