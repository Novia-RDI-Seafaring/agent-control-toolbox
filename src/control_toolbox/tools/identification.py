from io import FileIO
from control_toolbox.core import DataModel
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
        ...,
        description="The size of the input step that was applied in the experiment."
    )
    input_step_time: float = Field(
        ...,
        description="The time at which the input step was applied."
    )
    method: Literal["tangent", "smith", "s-k"] = Field(..., 
        description=(
            "Method to identify the model. Methods: "
            " 'tangent': Inflection-point method "
            " 'smith': Smith method"
            " 's-k': Sundaresan-Krishnaswamy method"
            )
        )
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

def identify_fopdt_from_step(data: DataModel, props: IdentificationProps) -> FOPDTModel:
    """
    Identifies a First Order Plus Dead Time (FOPDT) model from step response data.

    Estimates FOPDT model parameters (gain K, dead time L, time constant T) from
    step response data using one of three established identification methods. The
    function analyzes the output signal response to a known step input, extracting
    characteristic points to compute model parameters analytically.

    Purpose:
        Extract simplified FOPDT models from step response data for controller tuning
        and system analysis. FOPDT models are widely used in process control because
        they capture essential dynamics (gain, delay, time constant) while remaining
        simple enough for analytical controller design methods.

    Important:
        - Signal specified in output_name must exist in data, otherwise raises ValueError
        - Method must be one of: "tangent" (inflection point), "smith" (28%/63% points), "s-k" (35%/85% points)
        - System gain K is calculated as output_step_change / input_step_size
        - Dead time L and time constant T depend on the chosen identification method
        - Tangent method uses inflection point and tangent line intersection
        - Smith method uses 28.3% and 63.2% response points
        - S-K method uses 35.3% and 85.3% response points

    Args:
        data (DataModel):
            DataModel containing step response data with timestamps and output signals.
            Must contain the signal specified in props.output_name.
        props (IdentificationProps):
            Identification parameters including:
            - output_name: Name of the output signal to analyze
            - input_step_size: Magnitude of the input step applied in the experiment
            - input_step_time: Time when the input step was applied
            - method: Identification method - "tangent", "smith", or "s-k"
            - step_threshold: Threshold for step detection (default 1E-3)

    Returns:
        FOPDTModel:
            Identified FOPDT model containing:
            - K: System gain (process gain)
            - L: Dead time (time delay)
            - T: Time constant
            - description: Detailed description of the identification method and results
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
    t_step = props.input_step_time
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
            f"FOPDT model identified from step response using the tangent method. "
            f"It located the inflection point (t_i={t_i:.2f}, y_i={y_i:.2f}) of the "
            f"step response and uses the corresponding tangent line to estimate the "
            f"time constant T={T:.2f} and dead time L={L:.2f}. "
            f"The system gain is K={K:.2f} was estimated as the ratio of "
            f"step_change/input_step_size = {y_step:.2f}/{u_step:.2f} = {K:.2f}."
        )
        
    elif props.method == "smith":
        t28, y28 = _first_cross(t, y, 0.283 * y_inf)
        t63, y63 = _first_cross(t, y, 0.632 * y_inf)

        # system parameters
        T = 3/2 * (t63 - t28)
        L = t63 - T
        K = y_step / u_step if u_step != 0 else 0.0

        description = (
            "FOPDT model identified from step response using the Smiths method. It uses "
            f"characteristic points (t28={t28:.2f}, y28={y28:.2f}) and (t63={t63:.2f}, "
            f"y63={y63:.2f}) on the response curve to calculate the time constant "
            f"T={T:.2f} and dead time L={L:.2f} analytically."
            f" The system gain is K={K:.2f} was estimated as the ratio of "
            f"step_change/input_step_size = {y_step:.2f}/{u_step:.2f} = {K:.2f}."
        )

    elif props.method == "s-k":
        t35, y35 = _first_cross(t, y, 0.353 * y_inf)
        t85, y85 = _first_cross(t, y, 0.853 * y_inf)

        # system parameters
        T = 2/3 * (t85 - t35)
        L = 1.3 * t35 - 0.29 * t85
        K = y_step / u_step if u_step != 0 else 0.0

        description = (
            "FOPDT model identified from step response using the Sundaresan–Krishnaswamy (S–K) method. "
            f"It uses characteristic points (t35={t35:.2f}, y35={y35:.2f}) and (t85={t85:.2f}, "
            f"y85={y85:.2f}) on the response curve to calculate the time constant "
            f"T={T:.2f} and dead time L={L:.2f} analytically."
            f" The system gain is K={K:.2f} was estimated as the ratio of "
            f"step_change/input_step_size = {y_step:.2f}/{u_step:.2f} = {K:.2f}."
        )

    else:
        raise ValueError(f"Method '{props.method}' not supported")

    return FOPDTModel(
        K=K,
        L=L,
        T=T,
        description=description
    )

 
    
