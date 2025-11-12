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
    FOPDTModel,
    )
from control_toolbox.tools.pid_tuning import (
    zn_pid_tuning,
    UltimateTuningProps,
    UltimateGainParameters,
    lambda_tuning,
    LambdaTuningProps,
    PIDParameters,
    )
import numpy as np

FMU_DIR = "models/fmus/"
FMU_PATH = "models/fmus/PI_FOPDT_3.fmu"

## get fmu names
fmu_names = get_fmu_names(fmu_dir=FMU_DIR)
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
            "Kp": 3.84,
            "Ti": float("inf"),
        }
    )

# simulate step response
step_response = simulate_step_response(fmu_path=FMU_PATH, sim_props=simulation_props, step_props=step_props)

print(80*"=")
print("Simulated Step Response:")
print(step_response.model_dump_json(indent=2))
print(80*"=")

# find peaks
peak_props = FindPeaksProps()
peaks = find_peaks(data=step_response, props=peak_props)
print(80*"=")
print("Peaks:")
print(peaks.model_dump_json(indent=2))
print(80*"=")

# ultimate gain and period
ultimate_gain_props = UltimateGainParameters(
    Ku=3.84,
    Pu=peaks.attributes[0].average_peak_period
)
ultimate_tuning_props = UltimateTuningProps(
    params=ultimate_gain_props,
    controller="pid",
    method="classic"
)
# pid tuning
methods = ["classic", "some_overshoot", "no_overshoot"]

for method in methods:
    if method == "classic":
        controller = "pi"
        pid_parameters = zn_pid_tuning(
            props=UltimateTuningProps(
            params=ultimate_gain_props,
            controller=controller,
            method=method
        )
    )
    elif method == "some_overshoot":
        controller = "pid"
        pid_parameters = zn_pid_tuning(
            props=UltimateTuningProps(
                params=ultimate_gain_props,
                controller=controller,
                method=method
            )
        )

    elif method == "no_overshoot":
        controller = "pid"
        pid_parameters = zn_pid_tuning(
                props=UltimateTuningProps(
                params=ultimate_gain_props,
                controller=controller,
                method=method
            )
        )
    else:
        raise ValueError(f"Invalid method: {method}")

    print(80*"=")
    print(f"PID parameters for {method}:")
    print(pid_parameters.model_dump_json(indent=2))
    print(80*"=")


    # lambda tuning
    lambda_tuning_props = LambdaTuningProps(
        controller="pi",
        response="balanced"
    )
    pid_parameters = lambda_tuning(
        model=FOPDTModel(
            K=1.0,
            L=1.0,
            T=2.0,
            description="Test model"
        ),
        props=lambda_tuning_props
        )
    print(80*"=")
    print("Lambda tuning parameters:")
    print(pid_parameters.model_dump_json(indent=2))
    print(80*"=")
