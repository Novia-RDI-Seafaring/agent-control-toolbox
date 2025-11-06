from io import FileIO
from control_toolbox.core import DataModel, ResponseModel, Source
from pydantic import BaseModel, Field
from typing import Literal

class IdentifyFOPDTProps(BaseModel):
    input_name: str = Field(..., description="Name of the input signal")
    output_name: str = Field(..., description="Name of the output signal")
    method: Literal["tangent"] = Field(..., description="Method to identify the model")
    model: Literal["fopdt"] = Field(..., description="Model to identify")


########################################################
# TOOLS
########################################################

def identify_from_step(data: DataModel) -> ResponseModel:
    """
    Identify a FOPDT model from a step response.
    """
    return ResponseModel(
        source=Source(tool_name="identify_fopdt_tool"),
        data=data
    )