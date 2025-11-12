from control_toolbox.tools.simulation import simulate, simulate_step_response, SimulationProps, SimulationStepResponseProps
from control_toolbox.tools.signals import StepProps, TimeRange
from control_toolbox.tools.plotting import plot_data

FMU_DIR = "models/fmus/"
FMU_PATH = "models/fmus/PI_FOPDT_3.fmu"

# simulate
sim_props = SimulationProps(
    fmu_name="PI_FOPDT_3",
    start_time=0.0,
    stop_time=10.0,
    step_size=0.1,
    input=None,
    output=None,
    output_interval=0.1,
    start_values={
        "mode": False,
        "Kp": 1.0,
        "Ti": 1.0,
    }
)

step_props = StepProps(
    signal_name="input",
    time_range=TimeRange(start=0.0, stop=10.0, sampling_time=0.1),
    initial_value=0.0,
    final_value=1.0
)

results = simulate_step_response(fmu_path=FMU_PATH, sim_props=sim_props, step_props=step_props)
print(results.model_dump_json(indent=2))
print(80*"=")


# plot
from control_toolbox.tools.plotting import plot_data
