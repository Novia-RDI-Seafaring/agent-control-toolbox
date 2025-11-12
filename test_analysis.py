from control_toolbox.tools.information import (
    get_fmu_names,
    get_model_description
    )
from control_toolbox.tools.simulation import (
    simulate_step_response,
    SimulationStepResponseProps,
    SimulationProps
    )
from control_toolbox.tools.signals import (
    generate_step,
    StepProps,
    TimeRange,
    )
from control_toolbox.tools.analysis import (
    find_characteristic_points,
    find_peaks,
    FindPeaksProps,
    SettlingTimeProps,
    find_settling_time,
    find_rise_time,
    find_overshoot,
    FirstCrossingProps,
    find_first_crossing,
    InflectionPointProps,
    find_inflection_point
    )
from control_toolbox.tools.identification import (
    identify_fopdt_from_step,
    IdentificationProps,
    )
import numpy as np

FMU_DIR = "models/fmus/"
FMU_PATH = "models/fmus/PI_FOPDT_3.fmu"


## get fmu names
fmu_names = get_fmu_names(FMU_DIR)
print(fmu_names.model_dump_json(indent=2))
print(80*"=")

#get model desription
md = get_model_description(fmu_path=FMU_PATH)

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
        start_time=0.0,
        stop_time=20.0,
        output_interval=0.1,
        start_values={
            "mode": True,
            "Kp": 1.728,
            "Ti": 2.85,
        }
    )

# simulate step response
step_response = simulate_step_response(fmu_path=FMU_PATH, sim_props=simulation_props, step_props=step_props)

print(80*"=")
print("Simulated Step Response:")
print(step_response.model_dump_json(indent=2))
print(80*"=")

#  find characteristic points
characteristic_points = find_characteristic_points(data=step_response)
print(80*"=")
print("Characteristic Points:")
print(characteristic_points.model_dump_json(indent=2))
print(80*"=")

# find peaks
peaks = find_peaks(data=step_response, props=FindPeaksProps())
print(80*"=")
print("Peaks:")
print(peaks.model_dump_json(indent=2))
print(80*"=")

# find settling time
settling_time = find_settling_time(data=step_response, props=SettlingTimeProps())
print(80*"=")
print("Settling Time:")
print(settling_time.model_dump_json(indent=2))
print(80*"=")

# find rise time
rise_time = find_rise_time(data=step_response)
print(80*"=")
print("Rise Time:")
print(rise_time.model_dump_json(indent=2))
print(80*"=")

# find overshoot
overshoot = find_overshoot(data=step_response)
print(80*"=")
print("Overshoot:")
print(overshoot.model_dump_json(indent=2))
print(80*"=")
