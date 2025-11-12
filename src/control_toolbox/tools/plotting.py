from control_toolbox.tools.simulation import SimulationStepResponseProps, simulate_step_response
from control_toolbox.tools.signals import StepProps, TimeRange
from control_toolbox.core import DataModel
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


def plot_data(data: DataModel, show: bool = True) -> Figure:
    """
    Plots a DataModel.
    
    Args:
        data: DataModel containing timestamps and signals to plot
        show: Whether to display the plot (default: True)
    
    Returns:
        matplotlib figure object
    """
    # Configure plot style
    plt.rcParams.update({
        "text.usetex": False,               # Disable LaTeX (requires LaTeX installation)
        "font.family": "serif",             # Serif fonts
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],  # Fallback fonts
        "axes.labelsize": 12,               # Axis label font size
        "axes.titlesize": 12,               # (not used, but defined)
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 1.2,
        "axes.grid": False,
        "figure.dpi": 300,                  # High-resolution figure
        "savefig.dpi": 300,                 # Publication quality when saving
    })

    timestamps = np.array(data.timestamps)
    num_signals = len(data.signals)
    
    if num_signals == 0:
        raise ValueError("DataModel has no signals to plot")

    # Handle single subplot case (plt.subplots returns single Axes, not array)
    if num_signals == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3), sharex=True)
        axes = [ax]
    else:
        fig, axes = plt.subplots(num_signals, 1, figsize=(8, 3 * num_signals), sharex=True)

    for ax, s in zip(axes, data.signals):
        values = np.array(s.values)
        ax.plot(timestamps, values, label=s.name, color='black')
        ax.set_ylabel("value")
        ax.legend(loc='best')
        ax.margins(x=0)  # Remove x-axis margins to make x-axis tight, keep y-axis margins

    # Set xlabel on the last axis
    if num_signals == 1:
        axes[0].set_xlabel("Time")
    else:
        axes[-1].set_xlabel("Time")
    
    # --- Layout adjustments ---
    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(hspace=0.05)  # minimal vertical space

    if show:
        plt.show()
    
    return fig

if __name__ == "__main__":
    FMU_PATH = "models/fmus/PI_FOPDT_3.fmu"

    simulation_props = SimulationStepResponseProps(
        start_time=0.0,
        stop_time=20.0,
        output_interval=0.1,
        start_values={
            "mode": True,
            "Kp": 3.0,
            "Ti": float("inf"),
        }
    )
    step_props = StepProps(
        signal_name="input",
        time_range=TimeRange(start=0.0, stop=20.0, sampling_time=0.1),
        initial_value=0.0,
        final_value=1.0
    )
    results = simulate_step_response(fmu_path=FMU_PATH, sim_props=simulation_props, step_props=step_props)

    plot_data(results)

