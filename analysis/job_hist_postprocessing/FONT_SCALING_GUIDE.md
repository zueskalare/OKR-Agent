# Font Scaling Guide

## Problem Solved
Previously, the matplotlib template used fixed large font sizes (40pt, 50pt) that looked good on default-sized figures but became excessively large on bigger figures, causing poor visual balance.

## Solution: Intelligent Font Scaling

The template now features **automatic font scaling** that adjusts font sizes relative to figure dimensions, maintaining professional proportions regardless of figure size.

### Key Changes

1. **Base Font Sizes** (reduced from fixed large values):
   - Title: 20pt (was 40pt)
   - Axis labels: 16pt (was 50pt)
   - Tick labels: 14pt (was 40pt)
   - Legend: 14pt (was 40pt)
   - Colorbar: 16pt/14pt (was 40pt)

2. **Font Scale Factor**:
   - Applied as a multiplier to all base sizes
   - Default: 1.0 (100%)
   - Can be adjusted manually or automatically

3. **Auto-Scaling Algorithm**:
   - Calculates figure diagonal relative to default size (12×10 inches)
   - Uses sublinear scaling (square root) so fonts don't grow as fast as figures
   - Maintains visual balance across different figure sizes

## Usage Examples

### 1. Default Behavior (Auto-Scaling Enabled)
```python
# Fonts automatically scale based on figure size
fig, ax = create_publication_plot(
    xlabel="X Axis",
    ylabel="Y Axis",
    figsize=(16, 12)  # Larger figure → proportionally smaller fonts
)
```

### 2. Disable Auto-Scaling
```python
# Use base font sizes without auto-scaling
fig, ax = create_publication_plot(
    xlabel="X Axis",
    ylabel="Y Axis",
    figsize=(16, 12),
    auto_scale=False  # Fonts stay at base size
)
```

### 3. Manual Font Scaling
```python
# Manually adjust font scale
styler.set_font_scale(0.5)  # Half size fonts
fig, ax = create_publication_plot(xlabel="X", ylabel="Y")

styler.set_font_scale(1.5)  # 50% larger fonts
fig, ax = create_publication_plot(xlabel="X", ylabel="Y")

styler.set_font_scale(2.0)  # Double size fonts
fig, ax = create_publication_plot(xlabel="X", ylabel="Y")
```

### 4. Custom Initialization
```python
# Create a styler with a specific font scale
from matplotlib_layout import MatplotlibStyler

styler_small = MatplotlibStyler(font_scale=0.7)  # 70% size fonts
styler_large = MatplotlibStyler(font_scale=1.5)  # 150% size fonts
```

### 5. Programmatic Auto-Scaling
```python
# Manually trigger auto-scaling for a specific figure size
styler.auto_scale_fonts((20, 15))  # Scale fonts for a 20×15 inch figure

# Now create plots with those scaled fonts
fig, ax = styler.create_figure(figsize=(20, 15), auto_scale=False)
```

## Scaling Formula

The auto-scaling uses the following formula:

```
font_scale = sqrt(current_diagonal / default_diagonal)

where:
  default_diagonal = sqrt(12² + 10²) ≈ 15.62
  current_diagonal = sqrt(width² + height²)
```

**Examples:**
- (12, 10): scale = 1.00 (default)
- (6, 5): scale = 0.71 (small figure → smaller fonts)
- (24, 20): scale = 1.41 (large figure → moderately larger fonts)
- (48, 40): scale = 2.00 (very large figure → somewhat larger fonts)

The square root ensures fonts grow slower than the figure size, preventing them from dominating large plots.

## Benefits

1. **Automatic Balance**: Fonts always look proportional to the figure
2. **Flexibility**: Can override auto-scaling when needed
3. **Consistency**: All text elements scale together
4. **Publication Ready**: Works well for papers, posters, and presentations
5. **Backward Compatible**: Default behavior provides sensible results

## Migration from Old Version

If you were using the old template with fixed large fonts:

```python
# Old approach (fixed large fonts)
fig, ax = create_publication_plot(xlabel="X", ylabel="Y")
# → Fonts might be too large for big figures

# New approach (auto-scaled)
fig, ax = create_publication_plot(xlabel="X", ylabel="Y", figsize=(20, 16))
# → Fonts automatically adjust to figure size

# To get old behavior (large fixed fonts)
styler.set_font_scale(2.5)
fig, ax = create_publication_plot(xlabel="X", ylabel="Y", auto_scale=False)
```

## Best Practices

1. **Use auto-scaling by default** for most plots
2. **Disable auto-scaling** only when you need precise control
3. **Adjust font_scale** for specific use cases:
   - Posters: `font_scale=1.2-1.5`
   - Papers: `font_scale=1.0` (default)
   - Presentations: `font_scale=0.8-1.2` depending on figure size
4. **Test your plots** at the final output size
5. **Reload the module** after updates: `importlib.reload(matplotlib_layout)`

## Troubleshooting

**Problem**: Fonts still too large
```python
# Solution: Reduce font scale
styler.set_font_scale(0.7)
```

**Problem**: Fonts too small
```python
# Solution: Increase font scale
styler.set_font_scale(1.3)
```

**Problem**: Auto-scaling not working
```python
# Solution: Make sure auto_scale=True (default)
fig, ax = create_publication_plot(..., auto_scale=True)
```

**Problem**: Different font sizes for subplots
```python
# Solution: Disable auto-scaling and set manual scale
styler.set_font_scale(0.8)
fig, axes = styler.create_figure(2, 2, figsize=(16, 16), auto_scale=False)
```
