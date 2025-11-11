from pydantic import BaseModel, Field, model_validator
from typing import Any, Optional, Dict, Union, List
from datetime import datetime, timezone
from control_toolbox.storage import IDataStorage
########################################################
# DATA SCHEMA
########################################################
class Signal(BaseModel):
    name: str = Field(..., description="Name of the signal")
    values: List[float] = Field(..., description="List of values corresponding to the timestamps")
    unit: Optional[str] = Field(default=None, description="Unit of the signal")

class DataModelTeaser(BaseModel):
    timestamps: int = Field(..., description="Number of timestamps")
    signals: List[str] = Field(..., description="List of signal names")
    description: Optional[str] = Field(default=None, description="Description of the data")

class DataModel(BaseModel):
    timestamps: List[float] = Field(
        default_factory=list,
        description="List of timestamps"
    )
    signals: List[Signal] = Field(
        default_factory=list,
        description="List of signals, defined using the Signal schema"
    )
    description: Optional[str] = Field(default=None, description="Description of the data")

    @model_validator(mode="after")
    def check_length(self):
        """Ensure each signal has same number of values as timestamps."""
        tlen = len(self.timestamps)
        for s in self.signals:
            if len(s.values) != tlen:
                raise ValueError(
                    f"Length of timestamps and values in signal '{s.name}' "
                    f"does not match: timestamps={tlen}, values={len(s.values)}"
                )
        return self
    
    def to_teaser(self) -> DataModelTeaser:
        return DataModelTeaser(
            timestamps=len(self.timestamps),
            signals=[s.name for s in self.signals],
            description=self.description
        )

class AttributesGroup(BaseModel):
    title: str = Field(..., description="Title/name of the attribute group")
    attributes: List[Any] = Field(..., description="List of attributes")
    description: Optional[str] = Field(default=None, description="Description of the attribute group")

########################################################
# FIGURE SCHEMA
########################################################
class FigureModel(BaseModel):
    spec: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-friendly figure spec (e.g., Plotly dict)."
    )
    caption: Optional[str] = Field(None, description="Short human-readable caption for the figure.")
