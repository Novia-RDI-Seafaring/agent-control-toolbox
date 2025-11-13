from pathlib import Path
from typing import Dict, List, Optional, Union
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

def _get_fmu_information(fmu_path: Union[str, Path]) -> ModelDescription:
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

def get_fmu_names(fmu_folder: Union[str,Path]) -> FMUNamesResponse:
    """
    Lists all FMU model names in a directory.

    Scans a specified directory for FMU files (with '.fmu' extension) and returns
    their names without the file extension. This enables discovery of available
    simulation models in a workspace or model library.

    Purpose:
        Enable discovery and enumeration of available FMU simulation models in a
        workspace. Essential for building model selection interfaces and automated
        workflows that need to iterate over available models.

    Important:
        - Directory must exist and be accessible, otherwise raises FileNotFoundError
        - Only files with '.fmu' extension are included in results
        - Returns file stems (names without extension), not full paths
        - Empty list is returned if no FMU files are found in the directory

    Args:
        fmu_folder (Union[str, Path]):
            Path to the directory containing FMU model files. Must be a valid
            directory path, otherwise raises FileNotFoundError.

    Returns:
        FMUNamesResponse:
            Contains a list of FMU model names (file stems without '.fmu' extension)
            found in the specified directory.
    """
    
    if not isinstance(fmu_folder, Path):
        fmu_folder = Path(fmu_folder)

    if not fmu_folder.exists() or not fmu_folder.is_dir():
        raise FileNotFoundError(f"Invalid FMU directory: {fmu_folder}")

    fmu_names = [f.stem for f in fmu_folder.glob("*.fmu") if f.is_file()]
    return FMUNamesResponse(fmu_names=fmu_names)

def get_model_description(fmu_path: Union[str, Path]) -> ModelDescription:
    """
    Retrieves comprehensive model description from an FMU file.

    Reads and parses an FMU model file to extract metadata, variable definitions
    (inputs, outputs, parameters), default simulation options, and model information.
    This provides complete structural information about the model for analysis and
    simulation setup.

    Purpose:
        Extract complete structural and metadata information from FMU models to
        enable automated simulation setup, parameter discovery, and model analysis.
        Essential for understanding model capabilities and configuring simulations
        without manual inspection of FMU files.

    Important:
        - FMU file must exist and have '.fmu' extension, otherwise raises FileNotFoundError or ValueError
        - Uses FMPy's read_model_description to parse FMU XML metadata
        - Variable values are converted to floats with safe fallbacks for missing data
        - Default simulation options are extracted from FMU's defaultExperiment if available
        - Returns empty strings for missing metadata fields

    Args:
        fmu_path (Union[str, Path]):
            Path to the FMU file. Must end with '.fmu' extension and file must exist,
            otherwise raises ValueError or FileNotFoundError.

    Returns:
        ModelDescription:
            Complete model description containing:
            - model_name: Name of the FMU model
            - model_description: Text description of the model
            - variables: FMUVariables with lists of inputs, outputs, and parameters
            - metadata: FMUMetadata with FMI version, author, license, etc.
            - simulation: FMUSimulationOptions with default start/stop times, tolerance, step size
    """
    if not isinstance(fmu_path, Path):
        fmu_path = Path(fmu_path)

    # Check file extension
    if not fmu_path.suffix.lower() == ".fmu":
        raise ValueError(f"Invalid file extension: {fmu_path.name}. Expected a '.fmu' file.")

    # Check file existence
    if not fmu_path.exists():
        raise FileNotFoundError(f"FMU file not found: {fmu_path}")

    return _get_fmu_information(fmu_path)

