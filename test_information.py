from control_toolbox.tools.information import get_fmu_names, get_model_description
from pathlib import Path

FMU_DIR = "models/fmus/"
FMU_PATH = "models/fmus/PI_FOPDT_3.fmu"

# get fmu names
fmu_names = get_fmu_names(FMU_DIR)
print(fmu_names.model_dump_json(indent=2))
print(80*"=")

# get model description
model_description = get_model_description(FMU_PATH)
print(model_description.model_dump_json(indent=2))
print(80*"=")
