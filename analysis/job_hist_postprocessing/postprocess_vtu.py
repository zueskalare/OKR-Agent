#!/usr/bin/env python3
"""
PyVista-based postprocessing for VTU (VTK Unstructured Grid) files.

This script reads VTU files directly, which is a common output format
for FEniCSx, OpenFOAM, and other FEM/CFD software.

Requirements:
    pip install pyvista numpy

Usage:
    # Single VTU file
    python postprocess_vtu.py --vtu-file results.vtu

    # From job directory (auto-discovers VTU files)
    python postprocess_vtu.py --job-dir ../jobs/your_job

    # Deformed shape visualization
    python postprocess_vtu.py --vtu-file results.vtu --show-deformed --warp-scale 5.0

    # Specify output directory
    python postprocess_vtu.py --vtu-file results.vtu --output-dir ./my_plots
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import glob

try:
    import pyvista as pv
except ImportError:
    print("ERROR: This script requires pyvista.")
    print("Install with: pip install pyvista")
    sys.exit(1)


class VTUPostProcessor:
    """PyVista-based postprocessor for VTU files."""

    def __init__(
        self,
        vtu_file=None,
        job_dir=None,
        output_dir=None,
        resolution=(1920, 1080),
        show_deformed=False,
        warp_scale=1.0,
        displacement_field="u",
    ):
        """
        Initialize postprocessor.

        Args:
            vtu_file: Path to a single VTU file
            job_dir: Path to job directory (will search for VTU files)
            output_dir: Output directory for images
            resolution: Image resolution (width, height)
            show_deformed: Whether to show results on deformed shape
            warp_scale: Scale factor for deformation (only used if show_deformed=True)
            displacement_field: Name of displacement field for warping
        """
        self.vtu_files = []
        self.job_dir = None
        self.displacement_field = displacement_field

        if vtu_file:
            self.vtu_files = [Path(vtu_file).resolve()]
            self.job_dir = self.vtu_files[0].parent
        elif job_dir:
            self.job_dir = Path(job_dir).resolve()
            self._discover_vtu_files()
        else:
            raise ValueError("Either vtu_file or job_dir must be provided")

        if output_dir is None:
            self.output_dir = self.job_dir / "postprocess"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.resolution = resolution
        self.show_deformed = show_deformed
        self.warp_scale = warp_scale

        print(f"VTU files found: {len(self.vtu_files)}")
        for f in self.vtu_files:
            print(f"  - {f.name}")
        print(f"Output directory: {self.output_dir}")
        print(f"Show deformed shape: {self.show_deformed}")
        if self.show_deformed:
            print(f"Warp scale: {self.warp_scale}")

    def _discover_vtu_files(self):
        """Discover VTU files in job directory."""
        search_paths = [
            self.job_dir / "*.vtu",
            self.job_dir / "output" / "*.vtu",
            self.job_dir / "output" / "derived" / "*.vtu",
            self.job_dir / "results" / "*.vtu",
            self.job_dir / "**" / "*.vtu",  # Recursive search
        ]

        seen = set()
        for pattern in search_paths:
            for f in glob.glob(str(pattern), recursive=True):
                fpath = Path(f).resolve()
                if fpath not in seen:
                    self.vtu_files.append(fpath)
                    seen.add(fpath)

        if not self.vtu_files:
            print(f"WARNING: No VTU files found in {self.job_dir}")

    def _get_scalar_bar_args(self, title):
        """
        Get standardized scalar bar configuration.

        Args:
            title: Title for the colorbar

        Returns:
            dict: Scalar bar arguments for PyVista
        """
        return {
            "title": title,
            "title_font_size": 40,
            "label_font_size": 30,
            "font_family": "arial",
            "bold": True,
            "italic": False,
            "color": "black",
            "n_labels": 5,
            "fmt": "%.2e",  # Scientific notation with 2 decimals
            "vertical": True,
            "position_x": 0.85,
            "position_y": 0.25,
            "width": 0.08,
            "height": 0.5,
            "unconstrained_font_size": True,
        }

    def _add_custom_axes(self, plotter, position="lower_left", scale=0.25):
        """
        Add customized XYZ axes to the plotter with enhanced styling.

        Args:
            plotter: PyVista plotter object
            position: Position of axes widget
            scale: Scale factor for axes size
        """
        position_map = {
            "lower_left": (-0.03, -0.03),
            "lower_right": (0.65, 0.08),
            "upper_left": (0.08, 0.65),
            "upper_right": (0.65, 0.65),
        }

        viewport_pos = position_map.get(position, (-0.03, -0.03))

        plotter.add_axes(
            interactive=False,
            line_width=8,
            cone_radius=0.6,
            shaft_length=0.75,
            tip_length=0.25,
            ambient=0.6,
            label_size=(0.1, 0.1),
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
            x_color="red",
            y_color="green",
            z_color="blue",
            viewport=(*viewport_pos, scale, scale),
        )

    def load_vtu(self, vtu_path):
        """
        Load a VTU file using PyVista.

        Args:
            vtu_path: Path to VTU file

        Returns:
            pv.UnstructuredGrid: Loaded mesh with data
        """
        print(f"  Loading {vtu_path}...")
        grid = pv.read(str(vtu_path))

        # Print available arrays
        print(f"    Point arrays: {list(grid.point_data.keys())}")
        print(f"    Cell arrays: {list(grid.cell_data.keys())}")

        return grid

    def _get_vector_magnitude(self, grid, field_name):
        """
        Calculate magnitude of a vector field.

        Args:
            grid: PyVista grid
            field_name: Name of vector field

        Returns:
            numpy array of magnitudes
        """
        if field_name in grid.point_data:
            vec = grid.point_data[field_name]
        elif field_name in grid.cell_data:
            vec = grid.cell_data[field_name]
        else:
            raise KeyError(f"Field '{field_name}' not found in mesh")

        if vec.ndim == 1:
            return np.abs(vec)
        return np.linalg.norm(vec, axis=1)

    def _apply_deformation(self, grid, displacement_field=None):
        """
        Apply displacement field to deform a grid.

        Args:
            grid: PyVista grid
            displacement_field: Name of displacement field (default: self.displacement_field)

        Returns:
            Warped grid if show_deformed is True, else original grid
        """
        if not self.show_deformed:
            return grid

        disp_field = displacement_field or self.displacement_field

        if disp_field not in grid.point_data:
            print(
                f"  WARNING: Displacement field '{disp_field}' not found, skipping deformation"
            )
            return grid

        print(f"  Applying deformation with warp scale {self.warp_scale}...")
        warped = grid.warp_by_vector(disp_field, factor=self.warp_scale)
        return warped

    def _find_field(self, grid, possible_names, prefer_point_data=True):
        """
        Find a field by checking multiple possible names.

        Args:
            grid: PyVista grid
            possible_names: List of possible field names
            prefer_point_data: Whether to prefer point data over cell data

        Returns:
            tuple: (field_name, is_point_data) or (None, None)
        """
        data_sources = (
            [(grid.point_data, True), (grid.cell_data, False)]
            if prefer_point_data
            else [(grid.cell_data, False), (grid.point_data, True)]
        )

        for data, is_point in data_sources:
            for name in possible_names:
                if name in data:
                    return name, is_point

        return None, None

    def create_displacement_plot(self, grid, output_name="displacement_magnitude"):
        """Create displacement magnitude visualization."""
        print("\n=== Creating Displacement Plot ===")

        # Find displacement field
        disp_names = ["u", "U", "displacement", "Displacement", "disp", "d"]
        field_name, is_point = self._find_field(grid, disp_names)

        if field_name is None:
            print("  WARNING: No displacement field found, skipping")
            return

        # Calculate magnitude
        mag = self._get_vector_magnitude(grid, field_name)
        grid.point_data["Displacement_Magnitude"] = mag if is_point else None
        if not is_point:
            grid.cell_data["Displacement_Magnitude"] = mag
            grid = grid.cell_data_to_point_data()

        # Apply warp
        warped = grid.warp_by_vector(field_name, factor=self.warp_scale)
        warped.point_data["Displacement_Magnitude"] = grid.point_data[
            "Displacement_Magnitude"
        ]

        # Create plotter
        plotter = pv.Plotter(off_screen=True, window_size=self.resolution)
        plotter.add_mesh(
            warped,
            scalars="Displacement_Magnitude",
            cmap="viridis",
            show_edges=False,
            scalar_bar_args=self._get_scalar_bar_args("Displacement (m)"),
        )
        self._add_custom_axes(plotter, position="lower_left", scale=0.5)
        plotter.camera_position = "xy"
        plotter.background_color = "white"

        output_file = self.output_dir / f"{output_name}.png"
        plotter.screenshot(str(output_file))
        plotter.close()
        print(f"  Saved: {output_file}")

    def create_scalar_plot(
        self,
        grid,
        field_name,
        title,
        output_name,
        cmap="viridis",
        show_mesh=False,
        component=None,
    ):
        """
        Create a scalar field visualization.

        Args:
            grid: PyVista grid
            field_name: Name of field to visualize
            title: Title for colorbar
            output_name: Output filename (without extension)
            cmap: Colormap name
            show_mesh: Whether to show mesh edges
            component: If field is a vector/tensor, which component to extract
        """
        print(f"\n=== Creating {title} Plot ===")

        # Find the field
        is_point = field_name in grid.point_data
        is_cell = field_name in grid.cell_data

        if not is_point and not is_cell:
            print(f"  WARNING: Field '{field_name}' not found, skipping")
            return

        # Make a copy to avoid modifying original
        work_grid = grid.copy()

        # Extract component if needed
        scalars_name = field_name
        if component is not None:
            data = (
                work_grid.point_data[field_name]
                if is_point
                else work_grid.cell_data[field_name]
            )
            if data.ndim > 1 and component < data.shape[1]:
                comp_data = data[:, component]
                scalars_name = f"{field_name}_comp{component}"
                if is_point:
                    work_grid.point_data[scalars_name] = comp_data
                else:
                    work_grid.cell_data[scalars_name] = comp_data
            else:
                print(
                    f"  WARNING: Cannot extract component {component} from {field_name}"
                )
                return

        # Convert cell data to point data for smoother visualization
        if scalars_name in work_grid.cell_data:
            work_grid = work_grid.cell_data_to_point_data()

        # Apply deformation
        work_grid = self._apply_deformation(work_grid)

        # Create plotter
        plotter = pv.Plotter(off_screen=True, window_size=self.resolution)
        plotter.add_mesh(
            work_grid,
            scalars=scalars_name,
            cmap=cmap,
            show_edges=show_mesh,
            edge_color="black",
            line_width=0.5,
            scalar_bar_args=self._get_scalar_bar_args(title),
        )
        self._add_custom_axes(plotter, position="lower_left", scale=0.5)
        plotter.camera_position = "xy"
        plotter.background_color = "white"

        suffix = "_deformed" if self.show_deformed else ""
        output_file = self.output_dir / f"{output_name}{suffix}.png"
        plotter.screenshot(str(output_file))
        plotter.close()
        print(f"  Saved: {output_file}")

    def create_vonmises_plot(self, grid, output_name="von_mises_stress"):
        """Create von Mises stress visualization."""
        vm_names = [
            "sigma_vm",
            "von_mises",
            "vonMises",
            "VonMises",
            "sigma_von_mises",
            "stress_vm",
            "S_Mises",
            "Mises",
        ]
        field_name, _ = self._find_field(grid, vm_names, prefer_point_data=False)

        if field_name:
            self.create_scalar_plot(
                grid,
                field_name,
                "Von Mises Stress (Pa)",
                output_name,
                cmap="viridis",
            )
        else:
            print("\n=== Von Mises stress field not found, skipping ===")

    def create_stress_component_plots(self, grid):
        """Create plots for stress tensor components."""
        # Check for stress tensor field
        stress_names = ["sigma", "stress", "S", "Stress", "cauchy_stress", "S_Centroid"]
        field_name, is_point = self._find_field(grid, stress_names)

        if field_name is None:
            print("\n=== No stress tensor field found, skipping ===")
            return

        print(f"\n=== Creating Stress Component Plots (field: {field_name}) ===")

        components = [
            (0, "stress_xx", "Stress XX (Pa)"),
            (1, "stress_yy", "Stress YY (Pa)"),
            (3, "stress_xy", "Stress XY (Pa)"),
        ]

        for idx, name, title in components:
            self.create_scalar_plot(
                grid, field_name, title, name, cmap="viridis", component=idx
            )

    def create_strain_component_plots(self, grid):
        """Create plots for strain tensor components."""
        strain_names = ["epsilon", "strain", "E", "Strain", "eps", "E_Centroid"]
        field_name, is_point = self._find_field(grid, strain_names)

        if field_name is None:
            print("\n=== No strain tensor field found, skipping ===")
            return

        print(f"\n=== Creating Strain Component Plots (field: {field_name}) ===")

        components = [
            (0, "strain_xx", "Strain XX"),
            (1, "strain_yy", "Strain YY"),
            (3, "strain_xy", "Strain XY"),
        ]

        for idx, name, title in components:
            self.create_scalar_plot(
                grid, field_name, title, name, cmap="viridis", component=idx
            )

    def create_mesh_visualization(self, grid, output_name="mesh_structure"):
        """Create mesh structure visualization."""
        print("\n=== Creating Mesh Visualization ===")

        plotter = pv.Plotter(off_screen=True, window_size=self.resolution)
        plotter.add_mesh(
            grid,
            style="wireframe",
            color="black",
            line_width=1.0,
        )
        self._add_custom_axes(plotter, position="lower_left", scale=0.5)
        plotter.camera_position = "xy"
        plotter.background_color = "white"

        output_file = self.output_dir / f"{output_name}.png"
        plotter.screenshot(str(output_file))
        plotter.close()
        print(f"  Saved: {output_file}")

    def create_all_scalar_plots(self, grid):
        """Create plots for all scalar fields in the mesh."""
        print("\n=== Creating All Scalar Field Plots ===")

        # Combine point and cell data
        all_fields = []
        for name in grid.point_data.keys():
            data = grid.point_data[name]
            if data.ndim == 1:  # Scalar field
                all_fields.append((name, True))
        for name in grid.cell_data.keys():
            data = grid.cell_data[name]
            if data.ndim == 1:  # Scalar field
                all_fields.append((name, False))

        for field_name, is_point in all_fields:
            safe_name = field_name.replace(" ", "_").replace("/", "_")
            self.create_scalar_plot(
                grid,
                field_name,
                field_name,
                f"field_{safe_name}",
                cmap="viridis",
            )

    def process_single_vtu(self, vtu_path, prefix=""):
        """Process a single VTU file."""
        print(f"\n{'=' * 70}")
        print(f"Processing: {vtu_path.name}")
        print("=" * 70)

        grid = self.load_vtu(vtu_path)

        # Create standard plots
        self.create_mesh_visualization(grid, f"{prefix}mesh_structure")
        self.create_displacement_plot(grid, f"{prefix}displacement_magnitude")
        self.create_vonmises_plot(grid, f"{prefix}von_mises_stress")
        self.create_stress_component_plots(grid)
        self.create_strain_component_plots(grid)

    def run_all(self):
        """Run postprocessing on all discovered VTU files."""
        print("\n" + "=" * 70)
        print("VTU PyVista Post-Processing")
        print("=" * 70)

        if not self.vtu_files:
            print("ERROR: No VTU files to process")
            sys.exit(1)

        try:
            for i, vtu_file in enumerate(self.vtu_files):
                # Use prefix for multiple files
                prefix = f"{vtu_file.stem}_" if len(self.vtu_files) > 1 else ""
                self.process_single_vtu(vtu_file, prefix)

            print("\n" + "=" * 70)
            print("Post-processing completed successfully!")
            print(f"Output saved to: {self.output_dir}")
            print("=" * 70 + "\n")

        except Exception as e:
            print(f"\nERROR during post-processing: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Post-process VTU files using PyVista",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single VTU file
  python postprocess_vtu.py --vtu-file results.vtu

  # Process all VTU files in a job directory
  python postprocess_vtu.py --job-dir ../jobs/my_simulation

  # Show deformed shape with scale factor
  python postprocess_vtu.py --vtu-file results.vtu --show-deformed --warp-scale 10

  # Custom resolution
  python postprocess_vtu.py --vtu-file results.vtu --resolution 2560 1440
        """,
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--vtu-file", "-f", help="Path to a single VTU file")
    source_group.add_argument(
        "--job-dir", "-j", help="Path to job directory (will search for VTU files)"
    )

    parser.add_argument(
        "--output-dir", "-o", default=None, help="Output directory for images"
    )
    parser.add_argument(
        "--resolution",
        "-r",
        nargs=2,
        type=int,
        default=[1600, 1000],
        help="Image resolution (default: 1600 1000)",
    )
    parser.add_argument(
        "--field",
        choices=[
            "displacement",
            "vonmises",
            "stress",
            "strain",
            "mesh",
            "all",
            "all-scalars",
        ],
        default="all",
        help="Which fields to visualize",
    )
    parser.add_argument(
        "--show-deformed",
        action="store_true",
        help="Show stress/strain results on deformed shape",
    )
    parser.add_argument(
        "--warp-scale",
        type=float,
        default=1.0,
        help="Scale factor for displacement warping (default: 1.0)",
    )
    parser.add_argument(
        "--displacement-field",
        default="u",
        help="Name of displacement field for warping (default: 'u')",
    )
    parser.add_argument("--scalar-field", help="Name of specific scalar field to plot")
    parser.add_argument(
        "--cmap", default="viridis", help="Colormap for scalar plots (default: viridis)"
    )

    args = parser.parse_args()

    processor = VTUPostProcessor(
        vtu_file=args.vtu_file,
        job_dir=args.job_dir,
        output_dir=args.output_dir,
        resolution=tuple(args.resolution),
        show_deformed=args.show_deformed,
        warp_scale=args.warp_scale,
        displacement_field=args.displacement_field,
    )

    if not processor.vtu_files:
        print("ERROR: No VTU files found")
        sys.exit(1)

    # Load first VTU file for single-field operations
    grid = processor.load_vtu(processor.vtu_files[0])

    if args.scalar_field:
        # Plot a specific scalar field
        processor.create_scalar_plot(
            grid,
            args.scalar_field,
            args.scalar_field,
            f"field_{args.scalar_field}",
            cmap=args.cmap,
        )
    elif args.field == "all":
        processor.run_all()
    elif args.field == "all-scalars":
        processor.create_all_scalar_plots(grid)
    elif args.field == "displacement":
        processor.create_displacement_plot(grid)
    elif args.field == "vonmises":
        processor.create_vonmises_plot(grid)
    elif args.field == "stress":
        processor.create_stress_component_plots(grid)
    elif args.field == "strain":
        processor.create_strain_component_plots(grid)
    elif args.field == "mesh":
        processor.create_mesh_visualization(grid)


if __name__ == "__main__":
    main()
