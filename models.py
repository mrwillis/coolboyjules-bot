from pydantic import BaseModel, Field


class BinaryForecast(BaseModel):
    probability_percent: int = Field(ge=0, le=100, description="Probability as percentage (0-100)")
    reasoning: str = Field(description="Detailed reasoning for the forecast")


class PercentileValue(BaseModel):
    percentile: int = Field(description="Percentile (10, 20, 40, 60, 80, or 90)")
    value: float = Field(description="Value at this percentile")


class NumericForecast(BaseModel):
    percentiles: list[PercentileValue] = Field(description="List of percentile values in order")
    reasoning: str = Field(description="Detailed reasoning for the forecast")


class MultipleChoiceForecast(BaseModel):
    probabilities: dict[str, float] = Field(description="Probability for each option (should sum to 1.0)")
    reasoning: str = Field(description="Detailed reasoning for the forecast")
