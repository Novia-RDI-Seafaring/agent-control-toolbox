from plotly.graph_objs.pie import title
from pydantic import BaseModel, Field, model_validator
from control_toolbox.core import  DataModel, Signal, AttributesGroup
from typing import List, Tuple, Optional, Dict
import numpy as np
from datetime import datetime, timezone
from scipy.signal import find_peaks as scipy_find_peaks

########################################################
# SCHEMAS
########################################################
class TimeRange(BaseModel):
    start: float = Field(..., description="Start time of the time range")
    stop: float = Field(..., description="Stop time of the time range")
    sampling_time: float = Field(..., gt=0, description="Sampling time of the time range")

    @model_validator(mode="after")
    def _check_bounds(self):
        if self.stop <= self.start:
            raise ValueError("stop must be greater than start")
        return self
    
    @model_validator(mode="after")
    def _check_sampling_time(self):
        if self.sampling_time <= 0:
            raise ValueError("sampling_time must be greater than 0")
        return self

class StepProps(BaseModel):
    signal_name: str = Field(default="input", description="Name of the signal")
    time_range: TimeRange = Field(
        default_factory=lambda: TimeRange(start=0.0, stop=1.0, sampling_time=0.1),
        description="Time range over which the step signal is generated",
    )
    initial_value: float = Field(default=0.0, description="Initial value of the step signal")
    final_value: float = Field(default=1.0, description="Final value of the step signal")

    #@model_validator(mode="after")
    #def _check_step_time(self):
    #    tr = self.time_range
    #    if not (tr.start <= self.step_time <= tr.stop):
    #        raise ValueError("step_time must be within [start, stop]")
    #    return self

class ImpulseProps(BaseModel):
    """
    Properties for generating a discrete-time impulse (Dirac delta) signal.
    """
    signal_name: str = Field(
        default="input",
        description="Name of the signal carrying the impulse."
    )
    time_range: TimeRange = Field(
        default_factory=lambda: TimeRange(start=0.0, stop=1.0, sampling_time=0.1),
        description="Time range over which the impulse signal is defined."
    )
    impulse_time: float = Field(
        default=0.1,
        description="Time at which the unit impulse occurs (must fall within the time range)."
    )
    magnitude: float = Field(
        default=1.0,
        description="Amplitude of the impulse (default 1.0 for unit impulse)."
    )

    @model_validator(mode="after")
    def _check_impulse_time(self):
        tr = self.time_range
        if not (tr.start <= self.impulse_time <= tr.stop):
            raise ValueError("impulse_time must be within [start, stop]")
        return self

########################################################
# HELPER FUNCTIONS
########################################################


########################################################
# TOOLS
########################################################

def generate_step(step: StepProps) -> DataModel:
    """
    Generates a step signal with specified timing and amplitude.

    Creates a DataModel containing a step signal that transitions from an initial
    value to a final value at a specific time point. The signal is defined only
    at critical timestamps (start, step time, and end) to minimize data size
    while maintaining correct behavior in simulations.

    Purpose:
        Generate standardized step input signals for control system testing and
        simulation. Step signals are fundamental test inputs used to characterize
        system dynamics, generate step responses, and evaluate controller performance.

    Important:
        - Step transition occurs at time = time_range.start + time_range.sampling_time
        - Signal is defined only at three points: start, step time, and stop (minimal representation)
        - Signal name must match input variable name in FMU models when used for simulation
        - Timestamps are in ascending order and values list matches timestamps length
        - The minimal representation is sufficient for FMU simulation as it defines signal only where changes occur

    Args:
        step (StepProps):
            Step signal properties including:
            - signal_name: Name of the signal (default "input")
            - time_range: TimeRange with start, stop, and sampling_time
            - initial_value: Value before step transition (default 0.0)
            - final_value: Value after step transition (default 1.0)

    Returns:
        DataModel:
            Step signal containing timestamps and a single signal. The signal
            maintains initial_value until time_range.start + sampling_time,
            then transitions to final_value and remains constant until stop time.
    """
    t_start = step.time_range.start
    t_stop = step.time_range.stop
    dt = step.time_range.sampling_time

    v0 = step.initial_value
    v1 = step.final_value

    timestamps = [t_start, t_start + dt, t_stop]
    values = [v0, v1, v1]

    data = DataModel(
        timestamps=timestamps,
        signals=[Signal(name=step.signal_name, values=values)],
        description=f"Step signal on time interval [{t_start}, {t_stop}]. A step from {v0} to {v1} happens at t={t_start + dt}."
    )
    return data