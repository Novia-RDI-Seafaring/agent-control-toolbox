from pathlib import Path
from typing import Dict, List, Optional, Union
from fmpy import simulate_fmu as fmpy_simulate_fmu
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
    output: Optional[List[str]] = Field(
        default=None,
        description="Sequence of output variable names to record"
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
    output: Optional[List[str]] = Field(
        default=None,
        description="Sequence of output variable names to record"
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
    Simualtets with input defined in the SimulationProps.

    Args:
        fmu_path: str: Path to the FMU file. Must end with '.fmu'.
        sim_props: SimulationProps containing the simulation parameters

    Returns:
        DataModel: simulation results

    **Purpose:**  
    Run a time-domain simulation of a Functional Mock-up Unit (FMU) model using the specified parameters and input signals.

    **Important:**
    - Ensure that the output `output_interval` and the signal `sampling_time` are integer multiples of the FMU model step size (default 0.1).
    - Ensure that you have set all parameters correctly in the `start_values` dictionary before simulating.

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

    results = fmpy_simulate_fmu(
        filename=str(fmu_path),
        start_time=sim_props.start_time,
        stop_time=sim_props.stop_time,
        step_size=sim_props.step_size,
        start_values=sim_props.start_values,
        input=input_array,
        output=sim_props.output,
        output_interval=sim_props.output_interval,
        apply_default_start_values=True,
        record_events=True
    )

    data_model = ndarray_to_data_model(
        data=results,
        description=f"Simulation results for FMU in {fmu_path} with input {sim_props.input.description}"
        )
    
    return data_model

def simulate_step_response(fmu_path: Union[str, Path], sim_props: SimulationStepResponseProps, step_props: StepProps) -> DataModel:
    """
    Simualtets a step reponse with input defined in the StepProps.

    Args:
        fmu_path: str: Path to the FMU file. Must end with '.fmu'.
        sim_props: SimulationStepResponseProps containing the simulation parameters.
        step_props: StepProps containing the step signal properties.
        
    Returns:
        DataModel: step response of the FMU model.
    **Purpose:**  
    Simulate a step response of a Functional Mock-up Unit (FMU) model using the specified parameters and input signals.

    **Important:**
    - Ensure that the output `output_interval` and the signal `sampling_time` are integer multiples of the FMU model step size (default 0.1).
    - Ensure that you have set all parameters correctly in the `start_values` dictionary before simulating.

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
        output=sim_props.output,
        output_interval=sim_props.output_interval,
        start_values=sim_props.start_values
    )
    data_model = simulate(fmu_path, sim_props)
    data_model.description = f"""
    Simulated step response of FMU in {fmu_path} in time interval [{sim_props.start_time}, {sim_props.stop_time}].
    A step from {step_props.initial_value} to {step_props.final_value} happens at t={step_props.time_range.start + step_props.time_range.sampling_time}.
    """

    return data_model
