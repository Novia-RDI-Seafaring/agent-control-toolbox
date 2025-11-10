from control_toolbox.tools.information import get_fmu_names, get_model_description
from control_toolbox.tools.simulation import simulate, simulate_step_response, simulate_impulse_response, SimulationStepResponseProps, SimulationProps
from control_toolbox.tools.signals import (
    generate_step,
    StepProps, TimeRange,
    generate_impulse,
    ImpulseProps
    )
from control_toolbox.tools.analysis import (
    find_characteristic_points,
    find_peaks,
    FindPeaksProps,
    SettlingTimeProps,
    find_settling_time,
    FirstCrossingProps,
    find_first_crossing,
    InflectionPointProps,
    find_inflection_point
    )
from control_toolbox.tools.identification import (
    identify_fopdt_from_step,
    IdentificationProps,
    xcorr_analysis,
    XcorrProps
    )
import numpy as np

md = get_model_description(fmu_name="PI_FOPDT")

print(md.model_dump_json(indent=2))
print(80*"=")

###
step_props = StepProps(
    signal_name="input",
    time_range=TimeRange(start=0.0, stop=10.0, sampling_time=0.1),
    initial_value=0.0,
    final_value=1.0
)
step_results = generate_step(step_props)
print(80*"=")
print("Step Results:")
print(step_results.model_dump_json(indent=2))
print(80*"=")

# Create simulation properties
simulation_props = SimulationStepResponseProps(
        fmu_name="PI_FOPDT",
        start_time=0.0,
        stop_time=20.0,
        output_interval=0.1,
        start_values={
            "mode": False,
            "Kp": 1.0,
            "Ti": 1.0,
        }
    )

# simulate step response
step_response = simulate_step_response(sim_props=simulation_props, step_props=step_props)

print(80*"=")
print("Simulated Step Response:")
print(step_response.model_dump_json(indent=2))
print(80*"=")

###

identification_props = IdentificationProps(
    output_name="y",
    input_step_size=np.abs(step_props.final_value - step_props.initial_value),
    method="tangent",
    model="fopdt",
)
identification = identify_fopdt_from_step(step_response.data, props=identification_props)
print(80*"=")
print("Identification:")
print(identification.model_dump_json(indent=2))
print(80*"=")

