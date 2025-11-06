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
import numpy as np

class IdentificationProps(BaseModel):
    input_name: str = Field(..., description="Name of the input signal")
    output_name: str = Field(..., description="Name of the output signal")
    method: Literal["tangent", "smith", "s-k"] = Field(..., 
        description=(
            "Method to identify the model. Methods: "
            " 'tangent': Inflection-point method "
            " 'smith': Smith method"
            " 's-k': Sundaresan-Krishnaswamy method"
            )
        )
    model: Literal["fopdt"] = Field(..., description="Model to identify")

class FOPDTModel(BaseModel):
    K: float = Field(..., description="System gain")
    L: float = Field(..., description="Dead time")
    T: float = Field(..., description="Time constant")
    description: Optional[str] = Field(..., description="Description of the model")

########################################################
# TOOLS
########################################################

def identify_fopdt_from_step(data: DataModel, props: IdentificationProps) -> ResponseModel:
    """
    Identify a FOPDT model from a step response.
    """
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

    # detect where step changes
    t_step = t[np.where(np.diff(u) != 0)[0]]
    u_step = np.abs(u[0] - u[-1]) # step change of u
        
    y_inf = y[-1]
    y_0 = y[0]
    y_step = y_inf - y_0 # step change of y
    
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

    else:
        raise ValueError(f"Method '{props.method}' not supported")

    model = FOPDTModel(
        K=K,
        L=L,
        T=T,
        description=description
        )

    return ResponseModel(
        source=Source(tool_name="identify_fopdt_tool"),
        attributes=[AttributesGroup(
            title="FOPDT model identification results",
            attributes=[model],
            description = (
                "Identified FOPDT parameters from step response data: "
                "K (process gain), T (time constant), L (dead time)."
            )
            )]
    )