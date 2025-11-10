from io import FileIO
from control_toolbox.core import DataModel, ResponseModel, Source
from pydantic import BaseModel, Field
from typing import Literal, Optional
from control_toolbox.tools.analysis import (
    find_inflection_point, 
    InflectionPointProps,
    _get_inflection_point,
    _first_cross
    )
from control_toolbox.core import AttributesGroup
from control_toolbox.tools.information import get_model_description
import numpy as np

class IdentificationProps(BaseModel):
    #input_name: str = Field(..., description="Name of the input signal")
    output_name: str = Field(..., description="Name of the output signal")
    input_step_size: float = Field(
        default=1.0,
        description="The size of the input step that was applied in the experiment."
    )
    method: Literal["tangent", "smith", "s-k"] = Field(..., 
        description=(
            "Method to identify the model. Methods: "
            " 'tangent': Inflection-point method "
            " 'smith': Smith method"
            " 's-k': Sundaresan-Krishnaswamy method"
            )
        )
    model: Literal["fopdt"] = Field(..., description="Model to identify")
    step_threshold: Optional[float] = Field(
        default=1E-3,
        description=(
            "Threshold for step detection. Changes in input signal smaller than this "
            "value will be ignored. If None, uses 1% of the input signal range as default."
        )
    )

class FOPDTModel(BaseModel):
    K: float = Field(..., description="System gain")
    L: float = Field(..., description="Dead time")
    T: float = Field(..., description="Time constant")
    description: Optional[str] = Field(..., description="Description of the model")

########################################################
# TOOLS
########################################################
from typing import Dict
class IdentificationParameters(BaseModel):
    characteristic_points: Dict[str, float] = Field(..., description="Characteristic points of the step response")

def identify_fopdt_from_step(data: DataModel, props: IdentificationProps) -> ResponseModel:
    """
    Identify a First Order Plus Dead Time (FOPDT) model from step response data without input signal.
    """
    
    y = None
    t = np.asarray(data.timestamps, dtype=float)

    for s in data.signals:
        if s.name == props.output_name:
            y = np.asarray(s.values, dtype=float)

    if y is None:
        raise ValueError(f"Signal '{props.output_name}' not found in data")

    dt = data.timestamps[1] - data.timestamps[0]

    # time point when step is applied
    t_step = 0.0 + dt
    # input step size
    u_step = props.input_step_size
        
    y_inf = y[-1]
    y_0 = y[0]
    y_step = y_inf - y_0 # step change of y

    characteristic_points = {
        "y_inf": float(y_inf),
        "y_0": float(y_0),
        "y_step": float(y_step),
        "u_step": float(u_step),
        "t_step": t_step
    }
    
    if props.method == "tangent":

        # find inflection point
        t_i, y_i, slope = _get_inflection_point(t, y)
      
        # system gain
        L = t_i - (y_i / slope) - t_step if slope != 0 else float("inf") # point where tangent intersects x-axis -> dead time L
        T = t_i + (y_inf - y_i) / slope - t_step - L if slope != 0 else float("inf") # point where reaches the steady statevalue -> tiem constant T
        K = y_step / u_step if u_step != 0 else 0.0

        description = (
            "FOPDT model identified from step response using the tangent method. "
            "It locates the inflection point of the output response and uses the "#"
            "corresponding tangent line to estimate the time constant (T) and dead time (L)."
        )

        characteristic_points["t_i"] = float(t_i)
        characteristic_points["y_i"] = float(y_i)
        characteristic_points["slope"] = float(slope)
        
    elif props.method == "smith":
        t28, y28 = _first_cross(t, y, 0.283 * y_inf)
        t63, y63 = _first_cross(t, y, 0.632 * y_inf)

        # system parameters
        T = 3/2 * (t63 - t28)
        L = t63 - T
        K = y_step / u_step if u_step != 0 else 0.0

        description = (
            "FOPDT model identified from step response using the Smith method. "
            "It uses characteristic points on the response curve (typically at 28.3% and 63.2% "
            "of the total change) to calculate the time constant (T) and dead time (L) analytically."
        )

        characteristic_points["t28"] = float(t28)
        characteristic_points["y28"] = float(y28)
        characteristic_points["t63"] = float(t63)
        characteristic_points["y63"] = float(y63)

    elif props.method == "s-k":
        t35, y35 = _first_cross(t, y, 0.353 * y_inf)
        t85, y85 = _first_cross(t, y, 0.853 * y_inf)

        # system parameters
        T = 2/3 * (t85 - t35)
        L = 1.3 * t35 - 0.29 * t85
        K = y_step / u_step if u_step != 0 else 0.0

        description = (
            "FOPDT model identified from step response using the Sundaresan–Krishnaswamy (S–K) method. "
            "It uses the times at which the response reaches 35.3% and 85.3% of the total change "
            "to analytically estimate the time constant (T) and dead time (L)."
        )

        characteristic_points["t35"] = float(t35)
        characteristic_points["y35"] = float(y35)
        characteristic_points["t85"] = float(t85)
        characteristic_points["y85"] = float(y85)

    else:
        raise ValueError(f"Method '{props.method}' not supported")

    model = FOPDTModel(
        K=K,
        L=L,
        T=T,
        description=description
        )

    return ResponseModel(
        #source=Source(tool_name="identify_fopdt_tool"),
        attributes=[AttributesGroup(
            title="FOPDT model identification results",
            attributes=[model, characteristic_points],
            description = (
                "Identified FOPDT parameters from step response data: "
                "K (process gain), T (time constant), L (dead time)."
            )
            )]
    )


'''
def identify_fopdt_from_step(data: DataModel, props: IdentificationProps) -> ResponseModel:
    """
    Identify a First Order Plus Dead Time (FOPDT) model from step response data. 
    
    input_name (u) ──► [ MODEL ] ──► output_name (y)


    **Usage:**
    This tool analyzes a previously simulated or measured step response and fits a FOPDT model to it.

    **Inputs:**
    - `data`: A `DataModel` object that contains the step response signal to analyze.
      The `data` object should come directly from the output of a previous simulation or measurement tool,
      without any modification or reformatting.
    - `props`: A `IdentificationProps` object that contains the identification parameters.

    **Requirements:**
    - The `timestamps` and `values` arrays in each signal of the `DataModel` must be exactly the same length.
    - Do **not** resample, truncate the data from the previous tool before passing it here.
    - In the IdentificationProps, ensure that the input_name and output_name match the signal names in the DataModel.

    **Output:**
    Returns a `ResponseModel` containing the identified FOPDT parameters and diagnostic plots.
    """

    # get step-size of data
    u = None
    y = None
    t = np.asarray(data.timestamps, dtype=float)

    for s in data.signals:
        if s.name == props.input_name:
            u = np.asarray(s.values, dtype=float)
        elif s.name == props.output_name:
            y = np.asarray(s.values, dtype=float)

    if u is None:
        raise ValueError(f"Signal '{props.input_name}' not found in data")
    if y is None:
        raise ValueError(f"Signal '{props.output_name}' not found in data")

    # determine threshold for step detection
    if props.step_threshold is None:
        # default to 1% of signal range
        u_range = np.max(u) - np.min(u)
        threshold = 0.01 * u_range if u_range > 0 else 1e-10
    else:
        threshold = props.step_threshold

    # detect where step changes (only changes larger than threshold)
    t_step_indices = np.where(np.abs(np.diff(u)) > threshold)[0]
    t_step_array = t[t_step_indices] if len(t_step_indices) > 0 else np.array([], dtype=float)
    t_step = float(t_step_array[0]) if len(t_step_array) > 0 else float("nan")  # first step time
    u_step = np.abs(u[0] - u[-1]) # step change of u
        
    y_inf = y[-1]
    y_0 = y[0]
    y_step = y_inf - y_0 # step change of y

    characteristic_points = {
        "y_inf": float(y_inf),
        "y_0": float(y_0),
        "y_step": float(y_step),
        "u_step": float(u_step),
        "t_step": t_step,
        "t_step_all": t_step_array.tolist() if len(t_step_array) > 0 else [],  # all step times as list
    }
    
    if props.method == "tangent":

        # find inflection point
        t_i, y_i, slope = _get_inflection_point(t, y)
      
        # system gain
        L = t_i - (y_i / slope) - t_step if slope != 0 else float("inf") # point where tangent intersects x-axis -> dead time L
        T = t_i + (y_inf - y_i) / slope - t_step - L if slope != 0 else float("inf") # point where reaches the steady statevalue -> tiem constant T
        K = y_step / u_step if u_step != 0 else 0.0

        description = (
            "FOPDT model identified from step response using the tangent method. "
            "It locates the inflection point of the output response and uses the "#"
            "corresponding tangent line to estimate the time constant (T) and dead time (L)."
        )

        characteristic_points["t_i"] = float(t_i)
        characteristic_points["y_i"] = float(y_i)
        characteristic_points["slope"] = float(slope)
        
    elif props.method == "smith":
        t28, y28 = _first_cross(t, y, 0.283 * y_inf)
        t63, y63 = _first_cross(t, y, 0.632 * y_inf)

        # system parameters
        T = 3/2 * (t63 - t28)
        L = t63 - T
        K = y_step / u_step if u_step != 0 else 0.0

        description = (
            "FOPDT model identified from step response using the Smith method. "
            "It uses characteristic points on the response curve (typically at 28.3% and 63.2% "
            "of the total change) to calculate the time constant (T) and dead time (L) analytically."
        )

        characteristic_points["t28"] = float(t28)
        characteristic_points["y28"] = float(y28)
        characteristic_points["t63"] = float(t63)
        characteristic_points["y63"] = float(y63)

    elif props.method == "s-k":
        t35, y35 = _first_cross(t, y, 0.353 * y_inf)
        t85, y85 = _first_cross(t, y, 0.853 * y_inf)

        # system parameters
        T = 2/3 * (t85 - t35)
        L = 1.3 * t35 - 0.29 * t85
        K = y_step / u_step if u_step != 0 else 0.0

        description = (
            "FOPDT model identified from step response using the Sundaresan–Krishnaswamy (S–K) method. "
            "It uses the times at which the response reaches 35.3% and 85.3% of the total change "
            "to analytically estimate the time constant (T) and dead time (L)."
        )

        characteristic_points["t35"] = float(t35)
        characteristic_points["y35"] = float(y35)
        characteristic_points["t85"] = float(t85)
        characteristic_points["y85"] = float(y85)

    else:
        raise ValueError(f"Method '{props.method}' not supported")

    model = FOPDTModel(
        K=K,
        L=L,
        T=T,
        description=description
        )

    return ResponseModel(
        #source=Source(tool_name="identify_fopdt_tool"),
        attributes=[AttributesGroup(
            title="FOPDT model identification results",
            attributes=[model, characteristic_points],
            description = (
                "Identified FOPDT parameters from step response data: "
                "K (process gain), T (time constant), L (dead time)."
            )
            )]
    )
'''

class XcorrProps(BaseModel):
    x_name: str = Field(..., description="Name of the input signal")
    y_name: str = Field(..., description="Name of the output signal")

class CorrelationResults(BaseModel):
    correlation_coefficient: float = Field(..., description="Maximum cross-correlation value")
    lag_samples: int = Field(..., description="Lag (in samples) at which maximum correlation occurs")
    lag_time: float = Field(..., description="Lag (in time units) corresponding to lag_samples")

def xcorr_analysis(data: DataModel, props: XcorrProps) -> ResponseModel:
    """
    Perform cross-correlation analysis between two signals x and y.
    Identifies the lag (both in samples and time) that maximizes the cross-correlation.
    """
    x = y = None

    for s in data.signals:
        if s.name == props.x_name:
            x = np.asarray(s.values, dtype=float)
        elif s.name == props.y_name:
            y = np.asarray(s.values, dtype=float)

    if x is None:
        raise ValueError(f"Signal '{props.x_name}' not found in data")
    if y is None:
        raise ValueError(f"Signal '{props.y_name}' not found in data")

    # Compute full cross-correlation
    xcorr = np.correlate(x, y, mode='full')
    lags = np.arange(-len(x) + 1, len(y))

    # Find lag with maximum correlation
    lag_idx = np.argmax(xcorr)
    correlation_coefficient = float(xcorr[lag_idx])
    lag_samples = int(lags[lag_idx])

    # Convert lag in samples to time units (assumes uniform sampling)
    timestamps = np.asarray(data.timestamps)
    dt = float(np.mean(np.diff(timestamps)))  # average sampling interval
    lag_time = lag_samples * dt

    # Build response
    results = CorrelationResults(
        correlation_coefficient=correlation_coefficient,
        lag_samples=lag_samples,
        lag_time=lag_time
    )

    return ResponseModel(
        #source=Source(tool_name="xcorr_tool"),
        attributes=[AttributesGroup(
            title="Cross-Correlation Results",
            attributes=[results],
            description=(
                f"Cross-correlation between signals '{props.x_name}' and '{props.y_name}'. "
                f"Maximum correlation at lag = {lag_samples} samples ({lag_time:.3f} time units)."
            )
        )]
    )



 
    
