from pathlib import Path
from typing import Dict, List, Optional
from fmpy import read_model_description
from pydantic import BaseModel, Field
from control_toolbox.config import get_fmu_dir

########################################################
# SCHEMAS
########################################################

class ParameterModel(BaseModel):
    name: str = Field(description="Name of the variable")
    value: float = Field(description="Initial or default value of the variable")
    description: str = Field(description="Description of the variable")
    unit: str = Field(description="Unit of measurement for the variable")
    min: float = Field(description="Minimum allowed value for the variable")
    max: float = Field(description="Maximum allowed value for the variable")
    nominal: float = Field(description="Nominal value of the variable")
    unbounded: bool = Field(description="Whether the variable value is unbounded")

class FMUVariables(BaseModel):
    inputs: List[ParameterModel] = Field(default_factory=list, description="List of input variables")
    outputs: List[ParameterModel] = Field(default_factory=list, description="List of output variables")
    parameters: List[ParameterModel] = Field(default_factory=list, description="List of parameter variables")

class FMUMetadata(BaseModel):
    fmi_version: str = Field(description="FMI (Functional Mock-up Interface) version used by the model")
    author: str = Field(description="Author of the FMU model")
    version: str = Field(description="Version of the FMU model")
    license: str = Field(description="License information for the FMU model")
    generation_tool: str = Field(description="Tool used to generate the FMU model")
    generation_date_and_time: str = Field(description="Date and time when the FMU model was generated")

class FMUSimulationOptions(BaseModel):
    start_time: float = Field(description="Default simulation start time")
    stop_time: float = Field(description="Default simulation stop time")
    tolerance: float = Field(description="Default solver tolerance")
    step_size: float = Field(description="Default simulation step size (internal step size of the FMU model)")

class FMUInfo(BaseModel):
    name: str = Field(description="Name of the FMU model")
    relative_path: str = Field(description="Relative path to the FMU file")
    description: str = Field(description="Description of the FMU model")
    variables: FMUVariables = Field(description="Variables defined in the FMU model")
    metadata: FMUMetadata = Field(description="Metadata information about the FMU model")
    simulation: FMUSimulationOptions = Field(description="Default simulation options for the FMU model")

class FMUCollection(BaseModel):
    """Returns a collection of all available FMU models and their information."""
    fmus: Dict[str, FMUInfo] = Field(description="Dictionary mapping FMU model names to their information")

class ModelDescription(BaseModel):
    model_name: str = Field(description="Name of the FMU model")
    model_description: str = Field(description="Description of the FMU model")
    variables: FMUVariables = Field(description="Variables defined in the FMU model (inputs, outputs, parameters)")
    metadata: FMUMetadata = Field(description="Metadata information about the FMU model")
    simulation: FMUSimulationOptions = Field(description="Default simulation options for the FMU model")


########################################################
# HELPERS FUNCTIONS
########################################################

def _get_default_simulation_options(md):
    default_exp = md.defaultExperiment
    # Convert string values to float if needed
    start_time = float(default_exp.startTime) if default_exp and default_exp.startTime is not None else 0.0
    stop_time = float(default_exp.stopTime) if default_exp and default_exp.stopTime is not None else 0.0
    tolerance = float(default_exp.tolerance) if default_exp and default_exp.tolerance is not None else 1E-4
    step_size = float(default_exp.stepSize) if default_exp and default_exp.stepSize is not None else 0.1
    return FMUSimulationOptions(
        start_time=start_time,
        stop_time=stop_time,
        tolerance=tolerance,
        step_size=step_size
    )

def _get_fmu_information(fmu_path: str) -> ModelDescription:
    """
    Reads the FMU at fmu_path and returns a ModelDescription object
    containing variables, metadata, and simulation settings.
    """
    path = Path(fmu_path)
    md = read_model_description(str(path))

    def _create_parameter_model(v):
        """Convert a model variable to ParameterModel"""
        # Convert start value to float if it exists
        start_value = 0.0
        if v.start is not None:
            try:
                start_value = float(v.start)
            except (ValueError, TypeError):
                start_value = 0.0
        
        # Convert min/max/nominal to float if they exist
        min_val = 0.0
        max_val = 0.0
        nominal_val = 1.0
        if v.min is not None:
            try:
                min_val = float(v.min)
            except (ValueError, TypeError):
                min_val = 0.0
        if v.max is not None:
            try:
                max_val = float(v.max)
            except (ValueError, TypeError):
                max_val = 0.0
        if v.nominal is not None:
            try:
                nominal_val = float(v.nominal)
            except (ValueError, TypeError):
                nominal_val = 1.0
        
        return ParameterModel(
            name=v.name or '',
            value=start_value,
            description=v.description or '',
            unit=v.unit or '',
            min=min_val,
            max=max_val,
            nominal=nominal_val,
            unbounded=v.unbounded if hasattr(v, 'unbounded') else False
        )
    
    # Gather variables by causality and convert to ParameterModel
    input_vars = [_create_parameter_model(v) for v in md.modelVariables if v.causality == 'input']
    output_vars = [_create_parameter_model(v) for v in md.modelVariables if v.causality == 'output']
    parameter_vars = [_create_parameter_model(v) for v in md.modelVariables if v.causality == 'parameter']

    variables = FMUVariables(
        inputs=input_vars,
        outputs=output_vars,
        parameters=parameter_vars
    )

    # Metadata with safe fallbacks
    metadata = FMUMetadata(
        fmi_version=md.fmiVersion or '',
        author=md.author or '',
        version=md.version or '',
        license=md.license or '',
        generation_tool=md.generationTool or '',
        generation_date_and_time=md.generationDateAndTime or ''
    )

    # Simulation defaults with safe fallback for None
    simulation_description = _get_default_simulation_options(md)
    base_description = md.description or '' # get base description from FMU model

    return ModelDescription(
        model_name=md.modelName or '',
        model_description=base_description,
        variables=variables,
        metadata=metadata,
        simulation=simulation_description
    )

class FMUNamesResponse(BaseModel):
    fmu_names: List[str] = Field(description="List of available FMU model names")

########################################################
# TOOLS
########################################################

def get_fmu_names() -> List[str]:
    """Lists all FMU models in the directory.
       
    Returns:
        fmu_names: List[str]: List of model names (without .fmu extension)

    Purpose:
        Get names of all available FMU simulation models.
    """
    fmu_dir = get_fmu_dir()
    names = [f.stem for f in fmu_dir.glob("*.fmu") if f.is_file()]
    return FMUNamesResponse(fmu_names=names)

def get_model_description(fmu_name: str) -> ModelDescription:
    """Gets the model description of a specific FMU model.

    Args:
        fmu_name: str: Name of the FMU model (without .fmu extension)
       
    Returns:
        ModelDescription: The full model description object.

    Purpose:
        Get the model description of a model. Includes:
            - FMUVariables: Variables defined in the FMU model (inputs, outputs, parameters)
            - FMUMetadata: Metadata information about the FMU model
            - FMUSimulationOptions: Default simulation options for the FMU model
    """
    fmu_dir = get_fmu_dir()
    dir = str(fmu_dir / f"{fmu_name}.fmu")
    return _get_fmu_information(dir)

