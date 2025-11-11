import pytest
from pathlib import Path
from control_toolbox.config import set_fmu_dir
from control_toolbox.tools.information import get_model_description, get_fmu_names
from control_toolbox.tools.information import (
    FMUInfo, ParameterModel, FMUVariables, FMUMetadata, FMUSimulationOptions, 
    ModelDescription, FMUCollection
)

def test_get_fmu_names():
    fmu_names = get_fmu_names()
    assert fmu_names is not None
    assert len(fmu_names.fmu_names) > 0
    assert fmu_names.fmu_names == ["PI_FOPDT"]

def test_get_model_description():
    md = get_model_description("PI_FOPDT")
    assert md is not None
    assert md.model_name == "PI_FOPDT"
    assert md.model_description is not None
    assert md.variables is not None
    assert md.metadata is not None
    assert md.simulation is not None

    from devtools import debug
    variables  = md.variables
    assert len(variables.inputs) > 0
    assert len(variables.outputs) > 0
    assert len(variables.parameters) > 0