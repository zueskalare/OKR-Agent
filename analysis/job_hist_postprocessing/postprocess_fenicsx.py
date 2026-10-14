#!/usr/bin/env pvpython
"""
DEPRECATED — DO NOT USE. Use postprocess_pyvista.py or postprocess_vtu.py.

This ParaView/pvpython script has a defect that makes it actively misleading:
`GridStatus = ["Grid"]` does not match the grid name DOLFINx writes into the
XDMF, so the reader loads an EMPTY dataset (0 points, 0 cells) — and the
summary then prints a HARDCODED list of "Generated Visualizations", claiming
images that were never rendered. A real session shipped a summary.txt with
"Number of Points: 0" alongside 10 claimed images, 3 of which do not exist.

ParaView Python (pvpython) script for postprocessing FEniCSx elasticity results.

This script:
1. Loads displacement and stress fields from XDMF files
2. Generates high-quality visualizations:
   - Displacement magnitude with deformed mesh
   - Von Mises stress distribution
   - Stress components (σ_xx, σ_yy, σ_xy)
   - Strain components
3. Creates publication-quality images and animations
4. Exports data for further analysis

Usage:
    pvpython postprocess_fenicsx.py --job-dir <path_to_job_directory> [options]

Example:
    pvpython postprocess_fenicsx.py --job-dir ../jobs/session-atomagent-fem_demo0_3holes_full
    pvpython postprocess_fenicsx.py --job-dir ../jobs/session-atomagent-fem_demo0_3holes_full --resolution 1920 1080 --dpi 300
"""

import argparse
import os
import sys

# Hard deprecation gate: exit unless explicitly forced. See module docstring.
if os.environ.get("ATOMAGENTS_FORCE_PARAVIEW") != "1":
    sys.stderr.write(
        "postprocess_fenicsx.py is DEPRECATED (loads 0 points from DOLFINx "
        "XDMF and reports hardcoded success).\n"
        "Use instead:\n"
        "  python postprocess_pyvista.py --job-dir <dir>   (XDMF/H5)\n"
        "  python postprocess_vtu.py --job-dir <dir>       (VTU)\n"
        "Set ATOMAGENTS_FORCE_PARAVIEW=1 to run anyway.\n"
    )
    sys.exit(2)
from pathlib import Path

# ParaView imports
try:
    from paraview.simple import *
except ImportError:
    print("ERROR: This script requires ParaView's pvpython interpreter.")
    print("Please run with: pvpython postprocess_fenicsx.py")
    sys.exit(1)


class FEniCSXPostProcessor:
    """ParaView-based postprocessor for FEniCSx results."""

    def __init__(self, job_dir, output_dir=None, resolution=(1920, 1080), dpi=300):
        """
        Initialize postprocessor.

        Args:
            job_dir: Path to job directory containing output/ folder
            output_dir: Path to save images (default: job_dir/postprocess/)
            resolution: Image resolution (width, height)
            dpi: DPI for image export
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
        self.dpi = dpi

        # Check required files exist
        self._validate_files()

        print(f"Job directory: {self.job_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Resolution: {resolution[0]}x{resolution[1]}, DPI: {dpi}")

    def _validate_files(self):
        """Check that required XDMF files exist."""
        required_files = {
            "displacement": self.output_data_dir / "solution.xdmf",
            "von_mises": self.derived_dir / "field_sigma_vm.xdmf",
            "stress": self.derived_dir / "field_sigma.xdmf",
            "strain": self.derived_dir / "field_eps.xdmf",
        }

        missing = []
        for name, path in required_files.items():
            if not path.exists():
                missing.append(f"{name}: {path}")

        if missing:
            print("ERROR: Missing required files:")
            for m in missing:
                print(f"  - {m}")
            sys.exit(1)

    def setup_view(self, view=None):
        """Configure render view settings."""
        if view is None:
            view = GetActiveViewOrCreate("RenderView")

        view.ViewSize = self.resolution
        view.Background = [1, 1, 1]  # White background
        view.OrientationAxesVisibility = 0

        # Anti-aliasing and rendering settings
        # Use BackgroundColorMode for ParaView 5.12+ compatibility
        try:
            view.BackgroundColorMode = "Single Color"  # ParaView 5.12+
        except AttributeError:
            try:
                view.UseGradientBackground = 0  # ParaView 5.11 and earlier
            except AttributeError:
                pass  # Fallback if neither works

        return view

    def setup_colorbar(self, colorbar, title, orientation="Vertical"):
        """
        Configure colorbar with consistent styling.

        Args:
            colorbar: ParaView colorbar object
            title: Title for the colorbar
            orientation: 'Vertical' or 'Horizontal'
        """
        colorbar.Title = title
        colorbar.ComponentTitle = ""
        colorbar.TitleFontSize = 18
        colorbar.LabelFontSize = 14
        colorbar.TitleBold = 1
        colorbar.TitleItalic = 0
        colorbar.TitleColor = [0, 0, 0]  # Black text
        colorbar.LabelColor = [0, 0, 0]  # Black text
        colorbar.Orientation = orientation

        # Colorbar position and size
        if orientation == "Vertical":
            colorbar.Position = [0.88, 0.20]  # Right side
            colorbar.ScalarBarLength = 0.6
            colorbar.ScalarBarThickness = 16
        else:  # Horizontal
            colorbar.Position = [0.30, 0.05]  # Bottom center
            colorbar.ScalarBarLength = 0.4
            colorbar.ScalarBarThickness = 16

        # Number format for labels
        colorbar.RangeLabelFormat = "%.2e"  # Scientific notation with 2 decimals
        colorbar.AutoOrient = 0
        colorbar.Visibility = 1

        return colorbar

    def create_displacement_plot(self, warp_scale=10.0):
        """
        Create displacement magnitude visualization with deformed mesh.

        Args:
            warp_scale: Scale factor for deformation visualization
        """
        print("\n=== Creating Displacement Plot ===")

        # Load displacement field
        solution_xdmf = str(self.output_data_dir / "solution.xdmf")
        solution = XDMFReader(FileNames=[solution_xdmf])

        # Update pipeline first to ensure data is loaded
        solution.UpdatePipeline()

        # Enable displacement array
        solution.PointArrayStatus = ["u"]
        solution.UpdatePipeline()

        # Calculate displacement magnitude
        calculator = Calculator(Input=solution)
        calculator.ResultArrayName = "Displacement_Magnitude"
        calculator.Function = "mag(u)"  # 'u' is the displacement field name
        calculator.UpdatePipeline()

        # Apply warp by vector for deformed visualization
        warp = WarpByVector(Input=calculator)
        warp.Vectors = ["POINTS", "u"]
        warp.ScaleFactor = warp_scale
        warp.UpdatePipeline()

        # Create display
        view = self.setup_view()
        display = Show(warp, view)

        # Color by displacement magnitude
        ColorBy(display, ("POINTS", "Displacement_Magnitude"))

        # Get color transfer function
        lut = GetColorTransferFunction("Displacement_Magnitude")
        lut.ApplyPreset("Rainbow Desaturated", True)

        # Add color bar with custom settings
        colorbar = GetScalarBar(lut, view)
        self.setup_colorbar(colorbar, "Displacement (m)", orientation="Vertical")

        # Reset camera and render
        view.ResetCamera()
        Render()

        # Save image
        output_file = self.output_dir / "displacement_magnitude.png"
        SaveScreenshot(
            str(output_file),
            view,
            ImageResolution=self.resolution,
            TransparentBackground=0,
        )
        print(f"Saved: {output_file}")

        # Clean up
        Hide(warp, view)
        Delete(warp)
        Delete(calculator)
        Delete(solution)
        del lut, colorbar, display

    def create_vonmises_plot(self, show_mesh=True, mesh_color=[0, 0, 0]):
        """
        Create von Mises stress visualization.

        Args:
            show_mesh: Whether to overlay mesh edges
            mesh_color: RGB color for mesh edges
        """
        print("\n=== Creating Von Mises Stress Plot ===")

        # Load von Mises stress field
        vm_xdmf = str(self.derived_dir / "field_sigma_vm.xdmf")
        vm_field = XDMFReader(FileNames=[vm_xdmf])

        # Update pipeline first
        vm_field.UpdatePipeline()

        # Enable sigma_vm array
        vm_field.CellArrayStatus = ["sigma_vm"]
        vm_field.UpdatePipeline()

        # Convert cell data to point data for smoother visualization
        from paraview.simple import CellDatatoPointData

        cell_to_point = CellDatatoPointData(Input=vm_field)
        cell_to_point.UpdatePipeline()

        # Create display
        view = self.setup_view()
        display = Show(cell_to_point, view)

        # Color by von Mises stress (field name is 'sigma_vm')
        ColorBy(display, ("POINTS", "sigma_vm"))

        # Get color transfer function and set range
        lut = GetColorTransferFunction("sigma_vm")
        lut.ApplyPreset("Jet", True)

        # Update pipeline to get data range
        display.RescaleTransferFunctionToDataRange(True, False)

        # Show mesh edges if requested
        if show_mesh:
            display.SetRepresentationType("Surface With Edges")
            display.EdgeColor = mesh_color
            display.LineWidth = 1.0

        # Add color bar with custom settings
        colorbar = GetScalarBar(lut, view)
        self.setup_colorbar(colorbar, "Von Mises Stress (Pa)", orientation="Vertical")

        # Reset camera and render
        view.ResetCamera()
        Render()

        # Save image
        output_file = self.output_dir / "von_mises_stress.png"
        SaveScreenshot(
            str(output_file),
            view,
            ImageResolution=self.resolution,
            TransparentBackground=0,
        )
        print(f"Saved: {output_file}")

        # Also save with mesh overlay
        if not show_mesh:
            display.SetRepresentationType("Surface With Edges")
            display.EdgeColor = mesh_color
            Render()
            output_file_mesh = self.output_dir / "von_mises_stress_with_mesh.png"
            SaveScreenshot(
                str(output_file_mesh),
                view,
                ImageResolution=self.resolution,
                TransparentBackground=0,
            )
            print(f"Saved: {output_file_mesh}")

        # Clean up
        Hide(cell_to_point, view)
        Delete(cell_to_point)
        Delete(vm_field)
        del lut, colorbar, display

    def create_stress_component_plots(self):
        """Create individual plots for stress components (σ_xx, σ_yy, σ_xy)."""
        print("\n=== Creating Stress Component Plots ===")

        # Load stress tensor field
        stress_xdmf = str(self.derived_dir / "field_sigma.xdmf")
        stress_field = XDMFReader(FileNames=[stress_xdmf])

        # Update pipeline first
        stress_field.UpdatePipeline()

        # Enable sigma array
        stress_field.CellArrayStatus = ["sigma"]
        stress_field.UpdatePipeline()

        # Stress components to extract (assuming 2D plane stress/strain)
        # For 2D: sigma tensor has components [σ_xx, σ_yy, σ_zz, σ_xy, σ_yx, σ_xz, σ_zx, σ_yz, σ_zy]
        # Indices: 0=xx, 1=yy, 3=xy (symmetric, so xy=yx)
        components = [
            ("sigma_xx", 0, "Stress XX (Pa)"),
            ("sigma_yy", 1, "Stress YY (Pa)"),
            ("sigma_xy", 3, "Stress XY (Pa)"),
        ]

        view = self.setup_view()

        for comp_name, comp_idx, title in components:
            print(f"  Processing {comp_name}...")

            # Extract component using Calculator
            calculator = Calculator(Input=stress_field)
            calculator.ResultArrayName = comp_name
            # Extract specific component from 'sigma' tensor field
            calculator.Function = f"sigma[:,{comp_idx}]"
            calculator.UpdatePipeline()

            # Convert cell data to point data for smoother visualization
            from paraview.simple import CellDatatoPointData

            cell_to_point = CellDatatoPointData(Input=calculator)
            cell_to_point.UpdatePipeline()

            # Create display
            display = Show(cell_to_point, view)

            # Color by component (now at POINTS after conversion)
            ColorBy(display, ("POINTS", comp_name))

            # Get color transfer function
            lut = GetColorTransferFunction(comp_name)
            lut.ApplyPreset("Cool to Warm", True)

            # Update pipeline and rescale
            display.RescaleTransferFunctionToDataRange(True, False)

            # Add color bar with custom settings
            colorbar = GetScalarBar(lut, view)
            self.setup_colorbar(colorbar, title, orientation="Vertical")

            # Reset camera and render
            view.ResetCamera()
            Render()

            # Save image
            output_file = self.output_dir / f"stress_{comp_name}.png"
            SaveScreenshot(
                str(output_file),
                view,
                ImageResolution=self.resolution,
                TransparentBackground=0,
            )
            print(f"  Saved: {output_file}")

            # Clean up this iteration
            Hide(cell_to_point, view)
            Delete(cell_to_point)
            Delete(calculator)
            del lut, colorbar, display

        Delete(stress_field)

    def create_strain_component_plots(self):
        """Create individual plots for strain components (ε_xx, ε_yy, ε_xy)."""
        print("\n=== Creating Strain Component Plots ===")

        # Load strain tensor field
        strain_xdmf = str(self.derived_dir / "field_eps.xdmf")
        strain_field = XDMFReader(FileNames=[strain_xdmf])

        # Update pipeline first
        strain_field.UpdatePipeline()

        # Enable eps array
        strain_field.CellArrayStatus = ["eps"]
        strain_field.UpdatePipeline()

        # Strain components to extract (2D plane stress/strain)
        # For 2D: eps tensor has components [ε_xx, ε_yy, ε_zz, ε_xy, ε_yx, ε_xz, ε_zx, ε_yz, ε_zy]
        # Indices: 0=xx, 1=yy, 3=xy (symmetric, so xy=yx)
        components = [
            ("epsilon_xx", 0, "Strain XX"),
            ("epsilon_yy", 1, "Strain YY"),
            ("epsilon_xy", 3, "Strain XY"),
        ]

        view = self.setup_view()

        for comp_name, comp_idx, title in components:
            print(f"  Processing {comp_name}...")

            # Extract component using Calculator
            calculator = Calculator(Input=strain_field)
            calculator.ResultArrayName = comp_name
            # Extract specific component from 'eps' tensor field
            calculator.Function = f"eps[:,{comp_idx}]"
            calculator.UpdatePipeline()

            # Convert cell data to point data for smoother visualization
            from paraview.simple import CellDatatoPointData

            cell_to_point = CellDatatoPointData(Input=calculator)
            cell_to_point.UpdatePipeline()

            # Create display
            display = Show(cell_to_point, view)

            # Color by component (now at POINTS after conversion)
            ColorBy(display, ("POINTS", comp_name))

            # Get color transfer function
            lut = GetColorTransferFunction(comp_name)
            lut.ApplyPreset("Blue to Red Rainbow", True)

            # Update pipeline and rescale
            display.RescaleTransferFunctionToDataRange(True, False)

            # Add color bar with custom settings
            colorbar = GetScalarBar(lut, view)
            self.setup_colorbar(colorbar, title, orientation="Vertical")

            # Reset camera and render
            view.ResetCamera()
            Render()

            # Save image
            output_file = self.output_dir / f"strain_{comp_name}.png"
            SaveScreenshot(
                str(output_file),
                view,
                ImageResolution=self.resolution,
                TransparentBackground=0,
            )
            print(f"  Saved: {output_file}")

            # Clean up this iteration
            Hide(cell_to_point, view)
            Delete(cell_to_point)
            Delete(calculator)
            del lut, colorbar, display

        Delete(strain_field)

    def create_mesh_visualization(self):
        """Create a visualization showing just the mesh structure."""
        print("\n=== Creating Mesh Visualization ===")

        solution_xdmf = str(self.output_data_dir / "solution.xdmf")
        solution = XDMFReader(FileNames=[solution_xdmf])
        solution.GridStatus = ["Grid"]

        view = self.setup_view()
        display = Show(solution, view)

        # Show as wireframe
        display.SetRepresentationType("Wireframe")
        display.LineWidth = 1.0
        display.AmbientColor = [0, 0, 0]

        view.ResetCamera()
        Render()

        output_file = self.output_dir / "mesh_structure.png"
        SaveScreenshot(
            str(output_file),
            view,
            ImageResolution=self.resolution,
            TransparentBackground=0,
        )
        print(f"Saved: {output_file}")

        Delete(solution)

    def create_animation(self, field="von_mises", duration=5.0, fps=30):
        """
        Create rotation animation of a field.

        Args:
            field: Field to animate ('von_mises', 'displacement', etc.)
            duration: Animation duration in seconds
            fps: Frames per second
        """
        print(f"\n=== Creating Animation for {field} ===")

        # Load appropriate field
        if field == "von_mises":
            xdmf_file = str(self.derived_dir / "field_sigma_vm.xdmf")
            field_name = "f_16"
            title = "Von Mises Stress (Pa)"
            preset = "Jet"
        elif field == "displacement":
            xdmf_file = str(self.output_data_dir / "solution.xdmf")
            field_name = "f_15"
            title = "Displacement (m)"
            preset = "Rainbow Desaturated"
        else:
            print(f"Unknown field: {field}")
            return

        data = XDMFReader(FileNames=[xdmf_file])
        data.GridStatus = ["Grid"]

        view = self.setup_view()
        display = Show(data, view)

        if field == "displacement":
            calculator = Calculator(Input=data)
            calculator.ResultArrayName = "Displacement_Magnitude"
            calculator.Function = "mag(f_15)"
            display = Show(calculator, view)
            field_name = "Displacement_Magnitude"

        # Color by field
        ColorBy(display, ("POINTS", field_name))
        lut = GetColorTransferFunction(field_name)
        lut.ApplyPreset(preset, True)

        # Add color bar
        colorbar = GetScalarBar(lut, view)
        colorbar.Title = title
        colorbar.ComponentTitle = ""
        colorbar.TitleFontSize = 16
        colorbar.LabelFontSize = 14

        view.ResetCamera()

        # Create animation
        num_frames = int(duration * fps)
        animation_file = self.output_dir / f"{field}_animation.avi"

        print(f"Generating {num_frames} frames...")

        # Get camera
        camera = GetActiveCamera()

        for frame in range(num_frames):
            # Rotate camera around Z axis
            angle = 360.0 * frame / num_frames
            camera.Azimuth(360.0 / num_frames)
            Render()

            if frame % 30 == 0:
                print(f"  Frame {frame}/{num_frames}")

        print(f"Animation created (note: ParaView animation export requires GUI mode)")
        print(f"Consider using SaveAnimation() in GUI mode or individual frame export")

        Delete(data)

    def export_data_summary(self):
        """Export summary statistics from the simulation."""
        print("\n=== Exporting Data Summary ===")

        summary = {
            "job_directory": str(self.job_dir),
            "mesh_info": {},
            "field_ranges": {},
        }

        # Get mesh statistics
        solution_xdmf = str(self.output_data_dir / "solution.xdmf")
        solution = XDMFReader(FileNames=[solution_xdmf])
        solution.GridStatus = ["Grid"]
        solution.UpdatePipeline()

        mesh_info = solution.GetDataInformation()
        summary["mesh_info"] = {
            "num_points": mesh_info.GetNumberOfPoints(),
            "num_cells": mesh_info.GetNumberOfCells(),
        }

        # Get von Mises stress range
        vm_xdmf = str(self.derived_dir / "field_sigma_vm.xdmf")
        vm_field = XDMFReader(FileNames=[vm_xdmf])
        vm_field.GridStatus = ["Grid"]
        vm_field.UpdatePipeline()

        # Save summary
        summary_file = self.output_dir / "summary.txt"
        with open(summary_file, "w") as f:
            f.write("FEniCSx Simulation Post-Processing Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Job Directory: {summary['job_directory']}\n\n")
            f.write("Mesh Information:\n")
            f.write(f"  Number of Points: {summary['mesh_info']['num_points']}\n")
            f.write(f"  Number of Cells: {summary['mesh_info']['num_cells']}\n\n")
            f.write("Generated Visualizations:\n")
            f.write("  - displacement_magnitude.png\n")
            f.write("  - von_mises_stress.png\n")
            f.write("  - von_mises_stress_with_mesh.png\n")
            f.write("  - stress_sigma_xx.png\n")
            f.write("  - stress_sigma_yy.png\n")
            f.write("  - stress_sigma_xy.png\n")
            f.write("  - strain_epsilon_xx.png\n")
            f.write("  - strain_epsilon_yy.png\n")
            f.write("  - strain_epsilon_xy.png\n")
            f.write("  - mesh_structure.png\n")

        print(f"Saved: {summary_file}")

        Delete(vm_field)
        Delete(solution)

    def run_all(self, include_animations=False):
        """Run all postprocessing steps."""
        print("\n" + "=" * 70)
        print("FEniCSx ParaView Post-Processing")
        print("=" * 70)

        try:
            self.create_displacement_plot(warp_scale=1.0)
            self.create_vonmises_plot(show_mesh=False)
            self.create_stress_component_plots()
            self.create_strain_component_plots()
            self.create_mesh_visualization()

            if include_animations:
                self.create_animation(field="von_mises", duration=5.0)
                self.create_animation(field="displacement", duration=5.0)

            self.export_data_summary()

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
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Post-process FEniCSx elasticity results using ParaView Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pvpython postprocess_fenicsx.py --job-dir ../jobs/session-atomagent-fem_demo0_3holes_full
  pvpython postprocess_fenicsx.py --job-dir ../jobs/session-atomagent-fem_demo0_3holes_full --animations
  pvpython postprocess_fenicsx.py --job-dir ../jobs/session-atomagent-fem_demo0_3holes_full --resolution 3840 2160 --dpi 600
        """,
    )

    parser.add_argument(
        "--job-dir",
        "-j",
        required=True,
        help="Path to job directory containing output/ folder",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Output directory for images (default: job-dir/postprocess/)",
    )
    parser.add_argument(
        "--resolution",
        "-r",
        nargs=2,
        type=int,
        default=[1920, 1080],
        metavar=("WIDTH", "HEIGHT"),
        help="Image resolution (default: 1920 1080)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="DPI for image export (default: 300)"
    )
    parser.add_argument(
        "--animations",
        "-a",
        action="store_true",
        help="Generate rotation animations (slow)",
    )
    parser.add_argument(
        "--field",
        choices=["displacement", "von_mises", "stress", "strain", "all"],
        default="all",
        help="Which fields to visualize (default: all)",
    )

    args = parser.parse_args()

    # Initialize postprocessor
    processor = FEniCSXPostProcessor(
        job_dir=args.job_dir,
        output_dir=args.output_dir,
        resolution=tuple(args.resolution),
        dpi=args.dpi,
    )

    # Run requested processing
    if args.field == "all":
        processor.run_all(include_animations=args.animations)
    elif args.field == "displacement":
        processor.create_displacement_plot()
    elif args.field == "von_mises":
        processor.create_vonmises_plot()
    elif args.field == "stress":
        processor.create_stress_component_plots()
    elif args.field == "strain":
        processor.create_strain_component_plots()


if __name__ == "__main__":
    main()
