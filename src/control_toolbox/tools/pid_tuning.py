import numpy as np
from scipy.signal import find_peaks as scipy_find_peaks
from typing import List, Optional, Dict, Tuple, Union, Any
from pydantic import BaseModel, Field
from control_toolbox.core import DataModel, AttributesGroup
from typing import Literal

########################################################
# SCHEMAS
########################################################

class UltimateGainParameters(BaseModel):
    Ku: float = Field(..., description="Ultimate gain (K_u) defined as the maxumim controller gain (integral and derivative action disabled) at which the step response exhibits sustained oscillations.")
    Pu: float = Field(..., description="Ultimate period (P_u) defined as the period of the sustained oscillations.")

class UltimateTuningProps(BaseModel):
    params: UltimateGainParameters
    controller: Literal["p", "pi", "pd", "pid"]
    method: Literal["classic", "some_overshoot", "no_overshoot"] = "classic"

class PIDParameters(BaseModel):
    """
    PID controller parameters in ideal form.

    u(t) = Kp * [ e(t) + (1/Ti) ∫ e(t) dt + Td * de(t)/dt ]

    Notes:
    - Ti = ∞ disables the integral term.
    - Td = 0 disables the derivative term.
    """
    Kp: float = Field(default=1.0, description="Controller proportional gain.")
    Ti: float = Field(default=float("inf"), description="Controller integral time.")
    Td: float = Field(default=0.0, description="Controller derivative time.")


########################################################
# HELPER FUNCTIONS
########################################################

########################################################
# TOOLS FUNCTIONS
########################################################

def zn_pid_tuning(props: UltimateTuningProps) -> PIDParameters:
    """
    Compute PID controller parameters using the Ziegler-Nichols closed-loop
    (also called ultimate gain or continuous-cycling) tuning method.
    """
    Ku, Pu = props.params.Ku, props.params.Pu
    ctrl, method = props.controller, props.method

    match (ctrl, method):
        case ("p", "classic"):
            return PIDParameters(Kp=0.5 * Ku)
        case ("pi", "classic"):
            return PIDParameters(Kp=0.45 * Ku, Ti=Pu / 1.2)
        case ("pd", "classic"):
            return PIDParameters(Kp=0.8 * Ku, Td=Pu / 8.0)
        case ("pid", "classic"):
            return PIDParameters(Kp=0.6 * Ku, Ti=Pu / 2.0, Td=Pu / 8.0)
        case ("pid", "some_overshoot"):
            return PIDParameters(Kp=(1.0 / 3.0) * Ku, Ti=Pu / 2.0, Td=Pu / 3.0)
        case ("pid", "no_overshoot"):
            return PIDParameters(Kp=0.20 * Ku, Ti=Pu / 2.0, Td=Pu / 3.0)
        case _:
            raise ValueError(
                f"Unsupported controller/method combination: "
                f"controller='{ctrl}', method='{method}'"
            )
