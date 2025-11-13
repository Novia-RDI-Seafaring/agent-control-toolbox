import numpy as np
from typing import Optional
from control_toolbox.core import DataModel, Signal

def ndarray_to_data_model(data: np.ndarray, description: Optional[str] = None) -> DataModel:
    """
    Converts a structured numpy array from FMPy into a DataModel.

    Transforms simulation results from FMPy's structured numpy array format (with
    named fields for time and variables) into the DataModel format used throughout
    the control toolbox. This enables consistent data handling across analysis and
    visualization tools.

    Args:
        data (np.ndarray):
            Structured numpy array from FMPy simulation results. Must have a 'time'
            field and additional fields for each output variable. Raises ValueError
            if structure is invalid.
        description (Optional[str]):
            Optional description text for the DataModel. Defaults to None.

    Returns:
        DataModel:
            DataModel containing:
            - timestamps: List of time values from the 'time' field
            - signals: List of Signal objects, one for each variable field (excluding 'time')
            - description: Optional description string

    Purpose:
        Bridge between FMPy simulation output format and the control toolbox's
        DataModel format, enabling seamless integration of simulation results with
        analysis, plotting, and identification tools.

    Important:
        - Raises ValueError if array is not structured or missing 'time' field
        - All fields except 'time' are converted to Signal objects
        - Timestamps and signal values are converted to Python lists
        - Assumes all fields have the same length (standard for FMPy output)
    """
    if data.dtype.names is None or 'time' not in data.dtype.names:
        raise ValueError("Structured array must have a 'time' field.")

    timestamps = data['time'].tolist()

    signals = []
    for name in data.dtype.names:
        if name != 'time':
            signals.append(
                    Signal(name=name, values=data[name].tolist())
                )
    return DataModel(timestamps=timestamps, signals=signals)

def data_model_to_ndarray(input_model: Optional[DataModel]) -> Optional[np.ndarray]:
    """
    Converts a DataModel into a structured numpy array for FMPy simulation input.

    Transforms input signals from DataModel format into the structured numpy array
    format required by FMPy's simulate_fmu function. Creates a dtype with 'time'
    and one field per signal, ensuring proper format for FMU simulation.

    Args:
        input_model (Optional[DataModel]):
            DataModel containing timestamps and input signals, or None if no inputs.
            Must have non-empty timestamps, otherwise raises ValueError.

    Returns:
        Optional[np.ndarray]:
            Structured numpy array with dtype [('time', 'f8'), (signal_name, 'f8'), ...]
            and one row per timestamp. Returns None if input_model is None.

    Purpose:
        Bridge between the control toolbox's DataModel format and FMPy's input
        requirements, enabling use of generated signals (e.g., step signals) as
        inputs to FMU simulations.

    Important:
        - Returns None if input_model is None (no inputs)
        - Raises ValueError if timestamps list is empty
        - Raises ValueError if any signal has length different from timestamps
        - All signal values are converted to float64 ('f8') dtype
        - Array has one row per timestamp with columns for time and each signal
    """
    if input_model is None:
        return None
        
    # Extract timestamps and variable names
    timestamps = input_model.timestamps
    n = len(timestamps)
    if n == 0:
        raise ValueError("DataModel.timestamps is empty")

    # list singal names
    signal_names = [s.name for s in input_model.signals]

    # Define structured dtype
    dtype = [("time", "f8")] + [(name, "f8") for name in signal_names]

    # Prepare structured array
    arr = np.zeros(n, dtype=dtype)
    arr["time"] = np.asarray(timestamps, dtype=float)

    # Fill in each signal's values
    for s in input_model.signals:
        values = np.asarray(s.values, dtype=float)
        if len(values) != n:
            raise ValueError(
                f"Signal '{s.name}' length ({len(values)}) does not match timestamps ({n})"
            )
        arr[s.name] = values

    return arr
