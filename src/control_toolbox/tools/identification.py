from io import FileIO
from control_toolbox.core import DataModel, ResponseModel, Source
from pydantic import BaseModel, Field
from typing import Literal, Optional
from control_toolbox.tools.analysis import (
    find_inflection_point, 
    InflectionPointProps,
    _get_inflection_point
    )
from control_toolbox.core import AttributesGroup
import numpy as np

class IdentificationProps(BaseModel):
    input_name: str = Field(..., description="Name of the input signal")
    output_name: str = Field(..., description="Name of the output signal")
    method: Literal["tangent"] = Field(..., description="Method to identify the model")
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

    else:
        raise ValueError(f"Method '{props.method}' not supported")

    model = FOPDTModel(
        K=K,
        L=L,
        T=T,
        description=f"FOPDT model identified from step response"
        )

    return ResponseModel(
        source=Source(tool_name="identify_fopdt_tool"),
        attributes=[AttributesGroup(
            title="FOPDT model identification results",
            attributes=[model],
            description=f"FOPDT model identified from step response"
            )]
    )