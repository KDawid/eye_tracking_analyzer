from dataclasses import dataclass


@dataclass(frozen=True)
class Column:
    Timestamp = "timestamp"
    # Raw data
    LeftX = "left_x"
    LeftY = "left_y"
    RightX = "right_x"
    RightY = "right_y"
    LeftDiameter = "left_diameter"
    RightDiameter = "right_diameter"

    # Refined data
    EyeX = "eye_x"
    EyeY = "eye_y"
    Diameter = "diameter"

    # Fixation
    StartTimestamp = "start_timestamp"
    EndTimestamp = "end_timestamp"
    Duration = "duration"

    # Other columns
    SampleCount = "sample_count"
