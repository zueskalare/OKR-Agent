#!/usr/bin/env python3
"""
PyVista-based postprocessing for FEniCSx results.

This script uses PyVista (VTK Python wrapper) instead of ParaView's pvpython,
which is more stable for programmatic visualization.

Requirements:
    pip install pyvista meshio numpy

Usage:
    # Standard (undeformed)
    python postprocess_pyvista.py --job-dir ../jobs/your_job

    # Deformed shape at 1:1 scale
    python postprocess_pyvista.py --job-dir ../jobs/your_job --show-deformed

    # Exaggerated deformation (5x magnification)
    python postprocess_pyvista.py --job-dir ../jobs/your_job --show-deformed --warp-scale 5.0
"""

import argparse
import numpy as np
from pathlib import Path
import sys

try:
    import pyvista as pv
    import h5py
except ImportError:
    print("ERROR: This script requires pyvista and h5py.")
    print("Install with: pip install pyvista h5py")
    sys.exit(1)


class FEniCSxPostProcessor:
    """PyVista-based postprocessor for FEniCSx results."""

    def __init__(
        self,
        job_dir,
        output_dir=None,
        resolution=(1920, 1080),
        show_deformed=False,
        warp_scale=1.0,
    ):
        """
        Initialize postprocessor.

        Args:
            job_dir: Path to job directory
            output_dir: Output directory for images
            resolution: Image resolution (width, height)
            show_deformed: Whether to show results on deformed shape
            warp_scale: Scale factor for deformation (only used if show_deformed=True)
        """
        self.job_dir = Path(job_dir).resolve()
        self.output_data_dir = self.job_dir / "output"
        self.derived_dir = self.output_data_dir / "derived"

        if output_dir is None:
            self.output_dir = self.job_dir / "postprocess"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(exist_ok=True)
        self.resolution = resolution
        self.show_deformed = show_deformed
        self.warp_scale = warp_scale

        print(f"Job directory: {self.job_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Show deformed shape: {self.show_deformed}")
        if self.show_deformed:
            print(f"Warp scale: {self.warp_scale}")

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
            "position_x": 0.85,  # Right side, centered vertically
            "position_y": 0.25,
            "width": 0.08,
            "height": 0.5,
            "unconstrained_font_size": True,  # Allow custom font sizing
        }

    def _add_custom_axes(self, plotter, position="lower_left", scale=0.25):
        """
        Add customized XYZ axes to the plotter with enhanced styling.

        Args:
            plotter: PyVista plotter object
            position: Position of axes widget ('lower_left', 'lower_right', 'upper_left', 'upper_right')
            scale: Scale factor for axes size (default: 0.25)

        Customization features:
            - Color-coded axes: X=red, Y=green, Z=blue
            - Smaller cone to avoid blocking labels
            - More centered position
            - Larger size via scale parameter
            - Large Arial font labels for better visibility (0.12 units)
            - Enhanced lighting (ambient=0.6) for better visibility
            - Thicker lines (width=8) for prominence

        Returns:
            None (modifies plotter in-place)
        """
        # Position mapping for viewport coordinates - more centered
        position_map = {
            "lower_left": (-0.03, -0.03),  # More centered (was 0.002)
            "lower_right": (0.65, 0.08),  # More centered (was 0.72)
            "upper_left": (0.08, 0.65),  # More centered (was 0.72)
            "upper_right": (0.65, 0.65),  # More centered (was 0.72)
        }

        viewport_pos = position_map.get(position, (-0.03, -0.03))

        # Add axes with enhanced custom configuration
        plotter.add_axes(
            interactive=False,
            line_width=8,  # Thick lines for visibility
            cone_radius=0.6,  # Smaller cone to avoid blocking labels (was 0.8)
            shaft_length=0.75,  # Longer shaft for better proportion
            tip_length=0.25,  # Moderate tip length
            ambient=0.6,  # Better lighting
            label_size=(0.1, 0.1),  # Large labels
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
            x_color="red",
            y_color="green",
            z_color="blue",
            viewport=(*viewport_pos, scale, scale),  # Viewport size
        )

    def _load_h5_mesh(self, h5_path, field_name=None, is_cell_data=False):
        """Load mesh and optional field data from HDF5 file."""
        print(f"  Loading {h5_path}...")

        with h5py.File(h5_path, "r") as f:
            # Load geometry
            geometry = f["/Mesh/mesh/geometry"][:]
            topology = f["/Mesh/mesh/topology"][:]

            # Create points (add z-coordinate if 2D)
            if geometry.shape[1] == 2:
                points = np.column_stack([geometry, np.zeros(geometry.shape[0])])
            else:
                points = geometry

            # Create cells (triangles)
            cells = []
            for tri in topology:
                cells.append([3] + tri.tolist())

            # Create UnstructuredGrid
            grid = pv.UnstructuredGrid(
                cells, [pv.CellType.TRIANGLE] * len(topology), points
            )

            # Load field data if specified
            if field_name:
                try:
                    field_data = f[f"/Function/{field_name}/0"][:]
                    if is_cell_data:
                        grid.cell_data[field_name] = field_data
                    else:
                        grid.point_data[field_name] = field_data
                except KeyError:
                    print(f"  Warning: Field '{field_name}' not found in HDF5 file")

        return grid

    def _apply_deformation(self, grid):
        """Apply displacement field to deform a grid if show_deformed is True."""
        if not self.show_deformed:
            return grid

        print(f"  Applying deformation with warp scale {self.warp_scale}...")

        # Load displacement field
        solution_h5 = self.output_data_dir / "solution.h5"
        with h5py.File(solution_h5, "r") as f:
            displacement = f["/Function/u/0"][:]

        # Add displacement to grid point data
        grid.point_data["u"] = displacement

        # Warp the grid by displacement
        warped = grid.warp_by_vector("u", factor=self.warp_scale)

        # Remove the temporary displacement field
        if "u" not in grid.point_data or grid.point_data["u"] is displacement:
            warped.point_data.pop("u", None)

        return warped

    def create_displacement_plot(self, warp_scale=1.0):
        """Create displacement magnitude visualization."""
        print("\n=== Creating Displacement Plot ===")

        solution_h5 = self.output_data_dir / "solution.h5"
        grid = self._load_h5_mesh(solution_h5, field_name="u", is_cell_data=False)

        # Calculate displacement magnitude
        if "u" in grid.point_data:
            u = grid.point_data["u"]
            displacement_mag = np.linalg.norm(u, axis=1)
            grid.point_data["Displacement_Magnitude"] = displacement_mag

        # Apply warp
        warped = grid.warp_by_vector("u", factor=warp_scale)
        warped.point_data["Displacement_Magnitude"] = displacement_mag

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

        # Save image
        output_file = self.output_dir / "displacement_magnitude.png"
        plotter.screenshot(str(output_file))
        plotter.close()
        print(f"  Saved: {output_file}")

    def create_vonmises_plot(self, show_mesh=False):
        """Create von Mises stress visualization."""
        print("\n=== Creating Von Mises Stress Plot ===")

        vm_h5 = self.derived_dir / "field_sigma_vm.h5"
        grid = self._load_h5_mesh(vm_h5, field_name="sigma_vm", is_cell_data=True)

        # Convert cell data to point data for smoother visualization
        if "sigma_vm" in grid.cell_data:
            grid = grid.cell_data_to_point_data()

        # Apply deformation if requested
        grid = self._apply_deformation(grid)

        # Create plotter
        plotter = pv.Plotter(off_screen=True, window_size=self.resolution)
        plotter.add_mesh(
            grid,
            scalars="sigma_vm",
            cmap="viridis",
            show_edges=show_mesh,
            edge_color="black",
            line_width=0.5,
            scalar_bar_args=self._get_scalar_bar_args("Von Mises Stress (Pa)"),
        )
        self._add_custom_axes(plotter, position="lower_left", scale=0.5)
        plotter.camera_position = "xy"
        plotter.background_color = "white"

        # Save image
        suffix = "_deformed" if self.show_deformed else ""
        output_file = self.output_dir / f"von_mises_stress{suffix}.png"
        plotter.screenshot(str(output_file))
        plotter.close()
        print(f"  Saved: {output_file}")

        # Save with mesh overlay if not already shown
        if not show_mesh:
            plotter = pv.Plotter(off_screen=True, window_size=self.resolution)
            plotter.add_mesh(
                grid,
                scalars="sigma_vm",
                cmap="viridis",
                show_edges=True,
                edge_color="black",
                line_width=0.5,
                scalar_bar_args=self._get_scalar_bar_args("Von Mises Stress (Pa)"),
            )
            self._add_custom_axes(plotter, position="lower_left", scale=0.5)
            plotter.camera_position = "xy"
            plotter.background_color = "white"

            output_file_mesh = self.output_dir / "von_mises_stress_with_mesh.png"
            plotter.screenshot(str(output_file_mesh))
            plotter.close()
            print(f"  Saved: {output_file_mesh}")

    def create_stress_component_plots(self):
        """Create plots for stress components."""
        print("\n=== Creating Stress Component Plots ===")

        stress_h5 = self.derived_dir / "field_sigma.h5"
        grid = self._load_h5_mesh(stress_h5, field_name="sigma", is_cell_data=True)

        # Convert cell data to point data
        if "sigma" in grid.cell_data:
            grid = grid.cell_data_to_point_data()

        # Apply deformation if requested
        grid = self._apply_deformation(grid)

        # Stress tensor components (assuming 3x3 symmetric tensor stored as 9 components)
        # For 2D: indices are typically [xx, yy, zz, xy, yx, xz, zx, yz, zy]
        components = [
            (0, "sigma_xx", "Stress XX (Pa)", "viridis"),
            (1, "sigma_yy", "Stress YY (Pa)", "viridis"),
            (3, "sigma_xy", "Stress XY (Pa)", "viridis"),
        ]

        sigma = grid.point_data["sigma"]

        for idx, name, title, cmap in components:
            print(f"  Processing {name}...")

            # Extract component
            grid.point_data[name] = sigma[:, idx]

            # Create plotter
            plotter = pv.Plotter(off_screen=True, window_size=self.resolution)
            plotter.add_mesh(
                grid,
                scalars=name,
                cmap=cmap,
                show_edges=False,
                scalar_bar_args=self._get_scalar_bar_args(title),
            )
            self._add_custom_axes(plotter, position="lower_left", scale=0.5)
            plotter.camera_position = "xy"
            plotter.background_color = "white"

            # Save image
            suffix = "_deformed" if self.show_deformed else ""
            output_file = self.output_dir / f"stress_{name}{suffix}.png"
            plotter.screenshot(str(output_file))
            plotter.close()
            print(f"    Saved: {output_file}")

    def create_strain_component_plots(self):
        """Create plots for strain components."""
        print("\n=== Creating Strain Component Plots ===")

        strain_h5 = self.derived_dir / "field_eps.h5"
        grid = self._load_h5_mesh(strain_h5, field_name="epsilon", is_cell_data=True)

        # Convert cell data to point data
        if "epsilon" in grid.cell_data:
            grid = grid.cell_data_to_point_data()

        # Apply deformation if requested
        grid = self._apply_deformation(grid)

        components = [
            (0, "epsilon_xx", "Strain XX", "viridis"),
            (1, "epsilon_yy", "Strain YY", "viridis"),
            (3, "epsilon_xy", "Strain XY", "viridis"),
        ]

        eps = grid.point_data["epsilon"]

        for idx, name, title, cmap in components:
            print(f"  Processing {name}...")

            # Extract component
            grid.point_data[name] = eps[:, idx]

            # Create plotter
            plotter = pv.Plotter(off_screen=True, window_size=self.resolution)
            plotter.add_mesh(
                grid,
                scalars=name,
                cmap=cmap,
                show_edges=False,
                scalar_bar_args=self._get_scalar_bar_args(title),
            )
            self._add_custom_axes(plotter, position="lower_left", scale=0.5)
            plotter.camera_position = "xy"
            plotter.background_color = "white"

            # Save image
            suffix = "_deformed" if self.show_deformed else ""
            output_file = self.output_dir / f"strain_{name}{suffix}.png"
            plotter.screenshot(str(output_file))
            plotter.close()
            print(f"    Saved: {output_file}")

    def create_mesh_visualization(self):
        """Create mesh structure visualization."""
        print("\n=== Creating Mesh Visualization ===")

        solution_h5 = self.output_data_dir / "solution.h5"
        grid = self._load_h5_mesh(solution_h5)

        # Create plotter
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

        # Save image
        output_file = self.output_dir / "mesh_structure.png"
        plotter.screenshot(str(output_file))
        plotter.close()
        print(f"  Saved: {output_file}")

    def run_all(self):
        """Run all postprocessing steps."""
        print("\n" + "=" * 70)
        print("FEniCSx PyVista Post-Processing")
        print("=" * 70)

        try:
            self.create_displacement_plot(warp_scale=1.0)
            self.create_vonmises_plot(show_mesh=False)
            self.create_stress_component_plots()
            self.create_strain_component_plots()
            self.create_mesh_visualization()

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
        description="Post-process FEniCSx results using PyVista"
    )
    parser.add_argument("--job-dir", "-j", required=True, help="Path to job directory")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory")
    parser.add_argument(
        "--resolution",
        "-r",
        nargs=2,
        type=int,
        default=[1600, 1000],
        help="Image resolution (default: 1920 1080)",
    )
    parser.add_argument(
        "--field",
        choices=["displacement", "vonmises", "stress", "strain", "mesh", "all"],
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

    args = parser.parse_args()

    processor = FEniCSxPostProcessor(
        job_dir=args.job_dir,
        output_dir=args.output_dir,
        resolution=tuple(args.resolution),
        show_deformed=args.show_deformed,
        warp_scale=args.warp_scale,
    )

    if args.field == "all":
        processor.run_all()
    elif args.field == "displacement":
        processor.create_displacement_plot()
    elif args.field == "vonmises":
        processor.create_vonmises_plot()
    elif args.field == "stress":
        processor.create_stress_component_plots()
    elif args.field == "strain":
        processor.create_strain_component_plots()
    elif args.field == "mesh":
        processor.create_mesh_visualization()


if __name__ == "__main__":
    main()
