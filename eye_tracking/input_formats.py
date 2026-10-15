from dataclasses import dataclass, field
from typing import Literal

from eye_tracking.columns import Column


@dataclass(frozen=True)
class ValidityMapping:
    left: str | None = None
    right: str | None = None

    def as_dict(self) -> dict[str, str]:
        return {
            eye: column
            for eye, column in (("left", self.left), ("right", self.right))
            if column is not None
        }


@dataclass(frozen=True)
class RecordingFormat:
    columns: dict[str, str]
    time_unit: Literal["s", "ms", "us", "ns"] = "us"
    validity: ValidityMapping = field(default_factory=ValidityMapping)
    normalize: bool = False

    @property
    def read_columns(self) -> list[str]:
        return list(dict.fromkeys([*self.columns, *self.validity.as_dict().values()]))


TOBII = RecordingFormat(
    columns={
        "Time": Column.Timestamp,
        "L POR X [px]": Column.LeftX,
        "L POR Y [px]": Column.LeftY,
        "R POR X [px]": Column.RightX,
        "R POR Y [px]": Column.RightY,
        "L Mapped Diameter [mm]": Column.LeftDiameter,
        "R Mapped Diameter [mm]": Column.RightDiameter,
    },
    validity=ValidityMapping(left="L Validity", right="R Validity"),
    time_unit="us",
    normalize=True,
)

SIMPLEGAZETRACKER = RecordingFormat(
    columns={
        "Time": Column.Timestamp,
        "Lx": Column.LeftX,
        "Ly": Column.LeftY,
        "Rx": Column.RightX,
        "Ry": Column.RightY,
        "LPupil": Column.LeftDiameter,
        "RPupil": Column.RightDiameter,
    },
    time_unit="ms",
    normalize=True,
)
