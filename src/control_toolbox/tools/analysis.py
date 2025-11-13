import numpy as np
from scipy.signal import find_peaks as scipy_find_peaks
from typing import List, Optional, Dict, Tuple, Union, Literal, Any
from pydantic import BaseModel, Field
from control_toolbox.core import DataModel, AttributesGroup, Signal

########################################################
# SCHEMAS
########################################################
class Point(BaseModel):
    """
    A data point.
    """
    timestamp: float = Field(..., description="Timestamp of the data point.")
    value: float = Field(..., description="Value of the data point.")
    description: Optional[str] = Field(default=None, description="Description of the data point.")

class CharacteristicPoint(BaseModel):
    """
    Characteristic point.
    """
    name: str = Field(..., description="Name of the characteristic point.")
    description: str = Field(..., description="Description of the characteristic point.")
    point: Point = Field(..., description="Points of the characteristic point.")

class CharacteristicPoints(BaseModel):
    """
    Characteristic points of a step response.
    """
    signal_name: str = Field(..., description="Name of the signal.")
    characteristic_points: List[CharacteristicPoint] = Field(..., description="Points of the characteristic point.")

class FindPeaksProps(BaseModel):
    height: Optional[float] = Field(
        default=None,
        description=(
            "Required height of peaks. Either a number, None, an array matching x or a 2-element sequence of the former."
            "The first element is always interpreted as the minimal and the second, if supplied, as the maximal required height."
        )
    )
    threshold: Optional[float] = Field(
        default=None,
        description=(
            "Required threshold of peaks, the vertical distance to its neighboring samples."
            "Either a number, None, an array matching x or a 2-element sequence of the former."
            "The first element is always interpreted as the minimal and the second, if supplied, as the maximal required threshold."
        )
    )
    distance: Optional[float] = Field(
        default=None,
        description=(
            "Required distance between peaks. The minimum distance between returned peaks. "
            "Smaller peaks are removed first until the condition is fulfilled for all remaining peaks."
        )
    )
    prominence: Optional[float] = Field(
        default=None,
        description=(
            "Required prominence of peaks. Either a number, None, an array matching x or a 2-element sequence of the former."
            "The first element is always interpreted as the minimal and the second, if supplied, as the maximal required prominence."
        )
    )
    width: Optional[float] = Field(
        default=None,
        description=(
            "Required width of peaks in samples. Either a number, None, an array matching x or a 2-element sequence of the former."
            "The first element is always interpreted as the minimal and the second, if supplied, as the maximal required width"
        )
    )
    wlen: Optional[int] = Field(
        default=None,
        description=(
            "Used for calculation of the peaks prominences, thus it is only used if one of the arguments prominence or width is given."
            "See argument wlen in peak_prominences for a full description of its effects."
        )
    )
    rel_height: Optional[float] = Field(
        default=0.5,
        description=(
            "Used for calculation of the peaks width, thus it is only used if width is given."
            "See argument rel_height in peak_widths for a full description of its effects."
        )
    )
    plateau_size: Optional[int] = Field(
        default=None,
        description=(
            "Required size of the flat top of peaks in samples. Either a number, None, an array matching x or a 2-element sequence of the former."
            "The first element is always interpreted as the minimal and the second, if supplied, as the maximal required plateau size."
        )
    )

class Peaks(BaseModel):
    timestamps: List[float] = Field(..., description="Timestamps of detected peaks.")
    values: List[float] = Field(..., description="Values at detected peaks.")

class PeakAttributes(BaseModel):
    signal_name: str = Field(..., description="Name of the signal.")
    peaks: Peaks = Field(..., description="Detected peaks (timestamps and values).")
    average_peak_period: Optional[float] = Field(
        default=None, description="Average period between peaks (None if <2 peaks)."
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="scipy.signal.find_peaks properties."
    )

class TrendModel(BaseModel):
    signal_name: str = Field(..., description="Name of the signal.")
    slope: float = Field(..., description="Slope of the trend.")
    intercept: float = Field(..., description="Intercept of the trend.")
    status: Literal["increasing", "decreasing", "constant"] = Field(
        ..., description="Trend of the signal, i.e., is it increasing, decreasing, or constant."
    )
    description: Optional[str] = Field(
        default=None, description="Description of the trend."
    )

class OscillationAnalysisAttributes(BaseModel):
    signal_name: str = Field(..., description="Name of the signal.")
    peaks: Peaks = Field(..., description="Detected peaks (timestamps and values).")
    average_peak_period: Optional[float] = Field(
        default=None, description="Average period between peaks (None if <2 peaks)."
    )
    trend: TrendModel = Field(..., description="Trend of the peak amplitudes.")
    description: Optional[str] = Field(default=None, description="Per-signal summary.")

class FirstCrossingProps(BaseModel):
    signal_name: str = Field(..., description="Name of the signal.")
    threshold: float = Field(..., description="Threshold to detect.")
    start_index: int = Field(default=0, description="Index to start the search from.")
    is_upward: bool = Field(default=True, description="True for upward crossing (>=), False for downward crossing (<=).")

class InflectionPoint(BaseModel):
    signal_name: str = Field(..., description="Name of the signal.")
    timestamp: float = Field(..., description="Timestamp of the inflection point.")
    value: float = Field(..., description="Value of the inflection point.")
    slope: float = Field(..., description="Slope of the inflection point.")
    description: str = Field(..., description="Description of the inflection point.")
    
class InflectionPointProps(BaseModel):
    signal_name: str = Field(..., description="Name of the signal.")

class SettlingTimeProps(BaseModel):
    """
    Properties for finding steady state time point of a signal.
    """
    tolerance: float = Field(default=0.02, description="Tolerance for the steady state time point stays within a threshold (percentage) of steady-state value.")

class SettlingTime(BaseModel):
    signal_name: str = Field(..., description="Name of the signal.")
    settling_time: float = Field(..., description="Settling time of the signal.")

class RiseTime(BaseModel):
    signal_name: str = Field(..., description="Name of the signal.")
    rise_time: float = Field(..., description="Rise time of the signal.")
    description: str = Field(..., description="Description of the rise time. The rise time is the time it takes for the signal to rise from 10% to 90% of its final value.")

class Overshoot(BaseModel):
    signal_name: str = Field(..., description="Name of the signal.")
    max_value: float = Field(..., description="Maximum overshoot of the signal.")
    percent: float = Field(..., description="Percentage of the overshoot relative to steady state value.")
    description: str = Field(..., description="Description of the overshoot. The overshoot is the maximum deviation from the steady-state value after 90% of the change is reached.")

########################################################
# HELPER FUNCTIONS
########################################################

# Helper: get value at time (with interpolation)
def _get_value(timestamps: List[float], values: List[float], t_val: float) -> float:
    """
    Returns the interpolated value at time `t_val` using linear interpolation.

    - Uses numpy's `interp`, which handles interpolation and edge clipping.
    - Returns NaN for invalid inputs (e.g. mismatched or empty lists).

    Args:
        timestamps: Sorted list of time points.
        values: List of sample values corresponding to timestamps.
        t_val: Query time.

    Returns:
        Interpolated value, or NaN if input is invalid.
    """
    return float(np.interp(
        x=t_val,
        xp=timestamps,
        fp=values,
        left=values[0],      # Clip to first value when t_val < timestamps[0]
        right=values[-1]     # Clip to last value when t_val > timestamps[-1]
    ))

# Helper: first crossing of threshold
def _first_cross(
    timestamps: List[float],
    values: List[float],
    threshold: float,
    is_upward: bool = True,
    start_index: int = 0,
) -> Tuple[float, float]:
    """
    Returns the interpolated (t, y) after `start_index` where `values` crosses the given `threshold`.
    Uses linear interpolation between samples to find the exact crossing point.

    Args:
        timestamps: List of sample time points (same length as values).
        values: List of corresponding signal values.
        threshold: Threshold to detect.
        is_upward: True for upward crossing (>=), False for downward crossing (<=).
        start_index: Index to start the search from.

    Returns:
        (time, value) tuple at the first threshold crossing (interpolated), or (nan, nan) if none.
    """
    t = np.asarray(timestamps)
    y = np.asarray(values)

    # Find first index where threshold is crossed
    mask = (y[start_index:] >= threshold) if is_upward else (y[start_index:] <= threshold)
    idx = np.argmax(mask) if np.any(mask) else None

    if idx is None:
        return float("nan"), float("nan")
    
    i_cross = start_index + idx
    
    # If crossing at first point, can't interpolate backwards
    if i_cross == 0:
        return float(t[i_cross]), float(y[i_cross])
    
    # Interpolate between points before and at crossing
    t0, t1 = t[i_cross - 1], t[i_cross]
    y0, y1 = y[i_cross - 1], y[i_cross]
    
    # Linear interpolation: t_interp = t0 + (threshold - y0) * (t1 - t0) / (y1 - y0)
    dy = y1 - y0
    if np.abs(dy) > 1e-10:
        t_interp = t0 + (threshold - y0) * (t1 - t0) / dy
        return float(t_interp), float(threshold)
    else:
        return float(t[i_cross]), float(y[i_cross])

########################################################
# HELPER FUNCTIONS
########################################################

def _get_inflection_point(t, x):
    dxdt = np.diff(x)/np.diff(t)
    inflection_idx = np.argmax(dxdt)
    return t[inflection_idx], x[inflection_idx], dxdt[inflection_idx]

########################################################
# TOOLS FUNCTIONS
########################################################

def find_first_crossing(data: DataModel, props: FirstCrossingProps) -> AttributesGroup:
    """
    Finds the first crossing of a threshold in a signal.
    """
    points = []

    t = np.asarray(data.timestamps, dtype=float)
    signal_found = False
    for s in data.signals:
        if s.name == props.signal_name:
            signal_found = True
            x = np.asarray(s.values, dtype=float)
            tc, yc = _first_cross(t, x, props.threshold, props.is_upward, props.start_index)
            points.append(
                Point(
                    timestamp=tc,
                    value=yc,
                    description=f"Time point when signal {s.name} first reaches value {props.threshold:.2f}")
                )
            break
    
    if not signal_found:
        raise ValueError(f"Signal '{props.signal_name}' not found in data")

    attributes = AttributesGroup(
                title="First crossing results",
                attributes=points,
                description="First sample (t, y) after 'start_index' where 'values' crosses the given 'threshold'."
            )
    return attributes

def find_inflection_point(data: DataModel, props: InflectionPointProps) -> AttributesGroup:
    """
    Finds the inflection point of a signal.
    """
    points = []

    signal_found = False
    for s in data.signals:
        if s.name == props.signal_name:
            signal_found = True
            t_i, x_i, dxdt_i = _get_inflection_point(data.timestamps, s.values)
            points.append(
                InflectionPoint(
                    signal_name=s.name,
                    timestamp=t_i,
                    value=x_i,
                    slope=dxdt_i,
                    description=f"Inflection point of signal {s.name}")
                    )
            break
    
    if not signal_found:
        raise ValueError(f"Signal '{props.signal_name}' not found in data")

    attributes = AttributesGroup(
        title="Inflection point results",
        attributes=points,
        description=(
            "Returns the inflection point of a signal. In monotonic response curves "
            "like typical step responses, the inflection point is the location of "
            "maximum slope — where the rate of change is highest."
        ),
    )
    
    return attributes
    
def find_characteristic_points(data: DataModel) -> AttributesGroup:
    """
    Finds the characteristic points of step responses.
    
    Args:
        data: DataModel containing the signal
        
    Returns:
        ResponseModel: Contains **critical points* analyzing step rsponses and finetuning controllers.
            - p0 = (t0,y0) point when output starts to change from initial value.
            - p10 = (t10,y10) point when output first reachest 10% of total change.
            - p63 = (t63,y63) point when output first reachest 63% of total change. Can be used to determine the time constant T of a FOPDT system.
            - p90 = (t90,y90) point when output first reachest 90% of total change.
            - p98 = (t98,y98) point when output first reachest 98% of total change.
    """
    # find the points where the signal changes
    timestamps = data.timestamps

    characteristic_points = []
    for signal in data.signals:
        values = signal.values

        y_final = values[-1]
        t_final = timestamps[-1]
        
        # find the points where the signal changes
        t0, y0 = _first_cross(timestamps, values, 0.0)
        cp0 = CharacteristicPoint(
            name="p0",
            description="Point when output first starts to change from initial value.",
            point=Point(timestamp=t0, value=y0)
        )
        t10, y10 = _first_cross(timestamps, values, 0.1 * y_final)
        cp10 = CharacteristicPoint(
            name="p10",
            description="Point when output first reachest 10% of total change. Used as lower reference point when determining the rise time of a system.",
            point=Point(timestamp=t10, value=y10)
        )
        t63, y63 = _first_cross(timestamps, values, 0.63 * y_final)
        cp63 = CharacteristicPoint(
            name="p63",
            description="Point when output first reachest 63% of total change. Can be used to determine the time constant T of a FOPDT system.",
            point=Point(timestamp=t63, value=y63)
        )
        t90, y90 = _first_cross(timestamps, values, 0.90 * y_final)
        cp90 = CharacteristicPoint(
            name="p90",
            description="Point when output first reachest 90% of total change. Used as upper reference point when determining the rise time of a system.",
            point=Point(timestamp=t90, value=y90)
        )
        cp_final = CharacteristicPoint(
            name="pinf",
            description=f"Steady-state point of the step response as t to infty.",
            point=Point(timestamp=t_final, value=y_final)
        )

        cps = CharacteristicPoints(
            signal_name=signal.name,
            characteristic_points=[cp0, cp10, cp63, cp90, cp_final]
        )
        characteristic_points.append(cps)

    attributes = AttributesGroup(
        title="Characteristic points results",
        attributes=characteristic_points,
        description="Characteristic points of the step response."
    )

    return attributes

def find_peaks(data: DataModel, props: FindPeaksProps) -> AttributesGroup:
    """
    Find peaks inside each signal based on scipy.signal.find_peaks and
    return an AttributesGroup with per-signal PeakAttributes.
    """
    t = np.asarray(data.timestamps, dtype=float)

    peak_attributes: List[PeakAttributes] = []
    for signal in data.signals:
        x = np.asarray(signal.values, dtype=float)

        peaks_idx, properties = scipy_find_peaks(
            x,
            height=props.height,
            threshold=props.threshold,
            distance=props.distance,
            prominence=props.prominence,
            width=props.width,
            wlen=props.wlen,
            rel_height=props.rel_height,
            plateau_size=props.plateau_size,
        )

        peak_timestamps = [float(t[i]) for i in peaks_idx]
        peak_values = [float(x[i]) for i in peaks_idx]

        # Average peak period (None if fewer than 2 peaks)
        avg_period: Optional[float] = (
            float(np.mean(np.diff(peak_timestamps))) if len(peak_timestamps) >= 2 else None
        )

        # Make scipy properties JSON-serializable
        properties_serializable: Dict[str, Any] = {}
        for key, value in properties.items():
            if isinstance(value, np.ndarray):
                properties_serializable[key] = value.astype(float).tolist()
            elif isinstance(value, (np.floating, float, int)):
                properties_serializable[key] = float(value)
            else:
                properties_serializable[key] = value

        peak_attributes.append(
            PeakAttributes(
                signal_name=signal.name,
                peaks=Peaks(timestamps=peak_timestamps, values=peak_values),
                average_peak_period=avg_period,
                properties=properties_serializable,
            )
        )

    return AttributesGroup(
        title="Peak-detection results",
        attributes=peak_attributes,
        description="Detected peaks in all signals.",
    )

import warnings
def find_trend(data: DataModel, threshold: float = 0.02) -> List[TrendModel]:
    """
    Determines the linear trend of each signal in a DataModel.

    Classification is based on a dimensionless normalized slope:
        relative_slope = slope * (t_max - t_min) / mean(values)
    If |relative_slope| <= threshold -> 'constant'

    Args:
        data: DataModel containing the signals
        threshold: tolerance for 'constant' (default ±0.02 = ±2%)

    Returns:
        List[TrendModel] (one per signal)
    """
    t = np.asarray(data.timestamps, dtype=float)
    n = t.size

    if n < 2:
        raise ValueError("Need at least two samples to estimate trend")
    if np.allclose(t[-1], t[0]):
        raise ValueError("Timestamps have zero duration; cannot estimate slope")
    if n == 2:
        warnings.warn(
            "Trends computed from only two samples; results may be statistically unreliable.",
            UserWarning,
            stacklevel=2,
        )

    results: List[TrendModel] = []
    duration = t[-1] - t[0]

    for s in data.signals:
        y = np.asarray(s.values, dtype=float)
        slope, intercept = np.polyfit(t, y, 1)

        mean_val = float(np.mean(y))
        relative_slope = (slope * duration / mean_val) if mean_val != 0 else 0.0

        if relative_slope > threshold:
            status = "increasing"
        elif relative_slope < -threshold:
            status = "decreasing"
        else:
            status = "constant"

        desc = (
            f"Trend for '{s.name}' is {status}. "
            f"Slope={slope:.6g} (value per time), intercept={intercept:.6g}, "
            f"relative_slope={relative_slope:.3g} over the observed duration."
        )

        results.append(
            TrendModel(
                signal_name=s.name,
                slope=float(slope),
                intercept=float(intercept),
                status=status,
                description=desc,
            )
        )

    return results

def oscillation_analysis(data: DataModel) -> AttributesGroup:
    """
    Analyzes oscillations in a time series dataset:
      - Detects peaks per signal
      - Uses average peak period from peak detection (if available)
      - Computes trend of peak amplitudes (linear regression on peak values)
      - Packages per-signal results into OscillationAnalysisAttributes
      - Returns an AttributesGroup(title, attributes, description)
    """
    peak_result = find_peaks(data, props=FindPeaksProps())
    per_signal_attrs: List[OscillationAnalysisAttributes] = []

    for idx, s in enumerate(data.signals):
        peak_attr = peak_result.attributes[idx]  # assumes same ordering

        # NEW STRUCTURE: peaks.timestamps / peaks.values
        t_peaks = np.asarray(peak_attr.peaks.timestamps, dtype=float)
        v_peaks = np.asarray(peak_attr.peaks.values, dtype=float)

        # Average peak period: use provided value if present; otherwise compute
        if getattr(peak_attr, "average_peak_period", None) is not None:
            avg_period: Optional[float] = float(peak_attr.average_peak_period)
        else:
            avg_period = float(np.mean(np.diff(t_peaks))) if t_peaks.size >= 2 else None

        # Build a mini DataModel for trend on peak amplitudes (requires ≥2 samples)
        if t_peaks.size >= 2:
            peaks_data = DataModel(
                timestamps=t_peaks.tolist(),
                signals=[Signal(
                    name=s.name,
                    values=v_peaks.tolist(),
                    description=f"Detected peaks for '{s.name}'"
                )],
                description=f"Peaks extracted from '{s.name}'",
            )
            peak_trend = find_trend(peaks_data, threshold=0.05)[0]
        else:
            # Fallback when we don't have enough peaks to regress
            peak_trend = TrendModel(
                signal_name=s.name,
                slope=0.0,
                intercept=float(v_peaks[0]) if v_peaks.size == 1 else 0.0,
                status="constant",
                description=f"Insufficient peaks ({v_peaks.size}) to assess amplitude trend reliably.",
            )

        trend_to_behavior = {
            "increasing": "growing",
            "decreasing": "decaying",
            "constant": "sustained",
        }
        behavior = trend_to_behavior[peak_trend.status]

        # Refine trend description
        peak_trend.description = (
            f"The detected peak amplitudes of signal '{s.name}' are {peak_trend.status}. "
            f"Thus it exhibits *{behavior}* oscillations. "
            f"(peak slope {peak_trend.slope:.4g})"
        )

        per_signal_attrs.append(
            OscillationAnalysisAttributes(
                signal_name=s.name,
                peaks=Peaks(timestamps=t_peaks.tolist(), values=v_peaks.tolist()),
                average_peak_period=avg_period,
                trend=peak_trend,
                description=(
                    f"{s.name}: {len(v_peaks)} peaks; "
                    f"{'avg period=' + f'{avg_period:.6g}' if avg_period is not None else 'avg period unavailable'}; "
                    f"peak trend={peak_trend.status}."
                ),
            )
        )

    return AttributesGroup(
        title="Oscillation Analysis",
        attributes=per_signal_attrs,
        description=(
            f"Analyzed {len(data.signals)} signal(s). "
            "Computed peak timestamps/values, average period (if ≥2 peaks), and amplitude trend per signal."
        ),
    )


def find_settling_time(data: DataModel, props: SettlingTimeProps) -> AttributesGroup:
    """
    Finds the settling time of each signal in the data. The settling time is defined as the
    first time point where the signal remains within a specified tolerance (percentage) of
    its final value (i.e., steady-state level) for the remainder of the signal.
    """
    t = np.asarray(data.timestamps, dtype=float)
    if t.size == 0:
        raise ValueError("No timestamps in data")

    tol = props.tolerance
    settling_attributes = []

    for idx, s in enumerate(data.signals):
        x = np.asarray(s.values, dtype=float)
        steady_state = x[-1]

        # bounds for settling region
        ub = steady_state * (1 + tol)
        lb = steady_state * (1 - tol)

        # find point where signal stays within bounds
        within_band = (x >= lb) & (x <= ub)

        settling_time = float("nan")
        settling_value = float("nan")

        for i in range(len(within_band)):
            if within_band[i:].all():
                settling_time = float(t[i])
                settling_value = float(x[i])
                break

        # Add result as an Attribute entry
        settling_attributes.append(
            SettlingTime(
                signal_name=s.name,
                settling_time=settling_time
            )
        )

    attributes = AttributesGroup(
        title="Settling time results",
        attributes=settling_attributes,
        description="Settling times for each signal based on tolerance band."
    )
    return attributes

def find_rise_time(data: DataModel) -> AttributesGroup:
    """
    Finds the rise time of a signal. The rise time is the time it takes for the signal to rise from 10% to 90% of its final value.
    """
    rise_times = []
    for signal in data.signals:
        x = np.asarray(signal.values, dtype=float)
        t10, y10 = _first_cross(data.timestamps, x, 0.1 * x[-1])
        t90, y90 = _first_cross(data.timestamps, x, 0.90 * x[-1])
        rise_time = t90 - t10

        rise_times.append(
                RiseTime(
                signal_name=signal.name,
                rise_time=rise_time,
                description=f"Rise time of the signal {signal.name} is {rise_time:.2f}."
            )
        )

    attributes = AttributesGroup(
        title="Rise time of signals",
        attributes=rise_times,
        description="Rise time of the signals is definend as the time it takes for the signal to rise from 10% to 90% of its final value."
    )

    return attributes

def find_overshoot(data: DataModel) -> AttributesGroup:
    """
    Finds the maximum over- or undershoot of signal. The overshoot is the maximum deviation from the steady-state value after 90% of the change is reached.
    """
    overshoots = []
    t = np.asarray(data.timestamps, dtype=float)
    for signal in data.signals:
        x = np.asarray(signal.values, dtype=float)
        t90, y90 = _first_cross(data.timestamps, x, 0.90 * x[-1])
        
        # Find maximum value after t90
        mask = t >= t90
        max_value = float(np.max(x[mask]))
        overshoot = float(max_value - x[-1]) if np.any(mask) else 0.0
        percent = float(overshoot / x[-1] * 100)
        
        overshoots.append(
            Overshoot(
                signal_name=signal.name,
                max_value=max_value,
                percent=percent,
                description=f"Maximum overshoot of the signal {signal.name} is {overshoot:.2f} ({percent:.2f}%)."
            )
        )

    attributes = AttributesGroup(
        title="Overshoot of signals",
        attributes=overshoots,
        description="Overshoot of the signals"
    )
    return attributes


