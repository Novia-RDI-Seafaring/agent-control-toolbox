from pathlib import Path
from typing import Dict, List, Optional, Union
from fmpy import simulate_fmu as fmpy_simulate_fmu
from fmpy import read_model_description as fmpy_read_model_description
from pydantic import BaseModel, Field
from typing import Any

from control_toolbox.core import DataModel, FigureModel
from control_toolbox.tools.utils import data_model_to_ndarray, ndarray_to_data_model
from control_toolbox.config import get_fmu_dir
from control_toolbox.tools.signals import generate_step, StepProps

########################################################
# SCHEMAS
########################################################
class SimulationProps(BaseModel):
    start_time: Optional[Union[float, str]] = Field(
        default=0.0,
        description="Simulation start time"
    )
    stop_time: Optional[Union[float, str]] = Field(
        default=1.0,
        description="Simulation stop time"
    )
    step_size: Optional[Union[float, str]] = Field(
        default=None,
        description="Simulation step size. Must be integer multiple ofthe FMU models internal step size."
    )
    input: Optional[DataModel] = Field(
        default=None,
        description="DataModel containing input signals"
    )
    output_interval: Union[float, str] = Field(
        default=None,
        description="Interval for sampling the output. Must be integer multiple of FMU models internal step size."
    )
    start_values: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Dictionary of initial parameter and input values. "
            "Use this function to change the values of parameters and "
            "inputs from their default values."
        )
    )

class SimulationStepResponseProps(BaseModel):
    start_time: Optional[Union[float, str]] = Field(
        default=0.0,
        description="Simulation start time"
    )
    stop_time: Optional[Union[float, str]] = Field(
        default=1.0,
        description="Simulation stop time"
    )
    output_interval: Union[float, str] = Field(
        default=None,
        description="Interval for sampling the output. Must be integer multiple of FMU models internal step size."
    )
    start_values: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Dictionary of initial parameter and input values. "
            "Use this function to change the values of parameters and "
            "inputs from their default values."
        )
    )

########################################################
# PLOTTING
########################################################
import plotly.graph_objs as go

def plotly_simulation(data: DataModel):
    """
    Creates Plotly figures for simulation data visualization.

    Generates interactive Plotly line plots for each signal in a DataModel, creating
    separate figures for each signal with time on the x-axis and signal values on
    the y-axis. Figures are formatted with white plotly template for clean presentation.

    Purpose:
        Provide interactive visualization of simulation results for analysis and
        presentation. Plotly figures enable zooming, panning, and interactive
        exploration of control system responses.

    Important:
        - Creates a separate figure for each signal in the DataModel
        - Uses plotly_white template for consistent styling
        - X-axis is labeled "Time (seconds)", y-axis uses signal name
        - Figures are returned as FigureModel objects with Plotly dict specifications

    Args:
        data (DataModel):
            DataModel containing timestamps and signals to visualize.

    Returns:
        List[FigureModel]:
            List of FigureModel objects, one for each signal. Each figure contains
            a Plotly figure specification (as dictionary) and a caption describing
            the signal being plotted.
    """
    timestamps = data.timestamps
    signals = data.signals

    # Create a list to hold the individual figures
    figures = []

    for signal in signals:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=signal.values,
            mode='lines',
            name=signal.name
        ))
        fig.update_layout(
            title=f"Simulation Result - {signal.name}",
            xaxis_title="Time (seconds)",
            yaxis_title=f"{signal.name} Value",
            template="plotly_white"
        )
        # Convert Plotly figure to FigureModel
        figure_model = FigureModel(
            spec=fig.to_dict(),
            caption=f"Simulation result for {signal.name}"
        )
        figures.append(figure_model)

    return figures

########################################################
# TOOLS
########################################################
def simulate(fmu_path: Union[str, Path], sim_props: SimulationProps) -> DataModel:
    """
    Simulates a Functional Mock-up Unit (FMU) model with specified inputs and parameters.

    Executes a time-domain simulation of an FMU model using FMPy, allowing custom
    input signals, parameter values, and simulation settings. The function handles
    conversion between DataModel format and numpy arrays required by FMPy.

    Purpose:
        Enable time-domain simulation of control system models defined as FMU files.
        Essential for testing controller designs, analyzing system behavior, and
        generating step responses or other test signals for system identification.

    Important:
        - FMU file must exist and have '.fmu' extension, otherwise raises FileNotFoundError or ValueError
        - output_interval and input signal sampling_time must be integer multiples of FMU internal step size (default 0.1)
        - All parameters must be set correctly in start_values dictionary before simulating
        - Input DataModel is converted to structured numpy array format required by FMPy
        - Uses apply_default_start_values=True and record_events=True for comprehensive simulation

    Args:
        fmu_path (Union[str, Path]):
            Path to the FMU file. Must end with '.fmu' extension and file must exist.
        sim_props (SimulationProps):
            Simulation configuration including:
            - start_time: Simulation start time (default 0.0)
            - stop_time: Simulation stop time (default 1.0)
            - step_size: Internal simulation step size (must be integer multiple of FMU step size)
            - input: DataModel containing input signals (optional)
            - output_interval: Sampling interval for output (must be integer multiple of FMU step size)
            - start_values: Dictionary of initial parameter and input values (optional)

    Returns:
        DataModel:
            Simulation results containing timestamps and output signals. Description
            includes information about the FMU path and input signals used.
    """
    if not isinstance(fmu_path, Path):
        fmu_path = Path(fmu_path)

    # Check file extension
    if not fmu_path.suffix.lower() == ".fmu":
        raise ValueError(f"Invalid file extension: {fmu_path.name}. Expected a '.fmu' file.")

    # Check file existence
    if not fmu_path.exists():
        raise FileNotFoundError(f"FMU file not found: {fmu_path}")

    # check start values
    if sim_props.start_values is None:
        sim_props.start_values = {}
    

    # Convert DataModel input to numpy array if provided and not empty
    input_array = None
    if sim_props.input is not None and hasattr(sim_props.input, 'timestamps') and sim_props.input.timestamps:
        input_array = data_model_to_ndarray(sim_props.input)

    # model description
    md = fmpy_read_model_description(str(fmu_path))

    output_variables = [v.name for v in md.modelVariables if v.causality == 'output']

    results = fmpy_simulate_fmu(
        filename=str(fmu_path),
        start_time=sim_props.start_time,
        stop_time=sim_props.stop_time,
        step_size=sim_props.step_size,
        start_values=sim_props.start_values,
        input=input_array,
        output_interval=sim_props.output_interval,
        apply_default_start_values=True,
        record_events=True,
        output=output_variables # returns all output variables
    )

    data_model = ndarray_to_data_model(
        data=results,
        description=f"Simulation results for FMU in {fmu_path} with input {sim_props.input.description}"
        )
    
    return data_model

def simulate_step_response(fmu_path: Union[str, Path], sim_props: SimulationStepResponseProps, step_props: StepProps) -> DataModel:
    """
    Simulates a step response of a Functional Mock-up Unit (FMU) model.

    Generates a step input signal and runs a simulation to obtain the step response
    of an FMU model. This is a convenience function that combines step signal generation
    with simulation execution, commonly used for control system analysis and tuning.

    Purpose:
        Generate standardized step responses for control system analysis, controller
        tuning, and system identification. Step responses are fundamental test signals
        used to characterize system dynamics and evaluate controller performance.

    Important:
        - FMU file must exist and have '.fmu' extension, otherwise raises FileNotFoundError or ValueError
        - output_interval and step signal sampling_time must be integer multiples of FMU internal step size (default 0.1)
        - All parameters must be set correctly in start_values dictionary before simulating
        - Step occurs at time = time_range.start + time_range.sampling_time
        - The generated step signal is automatically passed to the simulate function

    Args:
        fmu_path (Union[str, Path]):
            Path to the FMU file. Must end with '.fmu' extension and file must exist.
        sim_props (SimulationStepResponseProps):
            Simulation configuration including:
            - start_time: Simulation start time (default 0.0)
            - stop_time: Simulation stop time (default 1.0)
            - output_interval: Sampling interval for output (must be integer multiple of FMU step size)
            - start_values: Dictionary of initial parameter and input values (optional)
        step_props (StepProps):
            Step signal properties including:
            - signal_name: Name of the input signal (default "input")
            - time_range: TimeRange with start, stop, and sampling_time
            - initial_value: Initial value before step (default 0.0)
            - final_value: Final value after step (default 1.0)

    Returns:
        DataModel:
            Step response results containing timestamps and output signals. Description
            includes information about step timing and magnitude.
    """
    if not isinstance(fmu_path, Path):
        fmu_path = Path(fmu_path)

    # Check file extension
    if not fmu_path.suffix.lower() == ".fmu":
        raise ValueError(f"Invalid file extension: {fmu_path.name}. Expected a '.fmu' file.")

    # Check file existence
    if not fmu_path.exists():
        raise FileNotFoundError(f"FMU file not found: {fmu_path}")

    # generate inputs
    input_step = generate_step(step_props)

    sim_props = SimulationProps(
        start_time=sim_props.start_time,
        stop_time=sim_props.stop_time,
        input=input_step,
        output_interval=sim_props.output_interval,
        start_values=sim_props.start_values
    )
    data_model = simulate(fmu_path, sim_props)
    data_model.description = f"""
    Simulated step response of FMU in {fmu_path} in time interval [{sim_props.start_time}, {sim_props.stop_time}].
    A step from {step_props.initial_value} to {step_props.final_value} happens at t={step_props.time_range.start + step_props.time_range.sampling_time}.
    """

    return data_model
