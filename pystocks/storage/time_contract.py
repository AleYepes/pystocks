from dataclasses import dataclass
from typing import Final, Literal

EndpointEffectiveAtPolicy = Literal[
    "endpoint_as_of_or_observed_date",
    "series_max_point_or_observed_date",
]


@dataclass(frozen=True)
class EndpointTimeContract:
    endpoint: str
    effective_at_policy: EndpointEffectiveAtPolicy
    source_as_of_description: str
    effective_at_description: str
    notes: str = ""


ENDPOINT_TIME_CONTRACTS: Final[dict[str, EndpointTimeContract]] = {
    "profile_and_fees": EndpointTimeContract(
        endpoint="profile_and_fees",
        effective_at_policy="endpoint_as_of_or_observed_date",
        source_as_of_description=(
            "No stable endpoint-level as_of_date is guaranteed; field-level dates such "
            "as total_net_assets_date remain separate facts."
        ),
        effective_at_description=(
            "Use a top-level endpoint as_of_date when present, otherwise the "
            "observation date."
        ),
        notes="Do not anchor profile snapshots to ratios.as_of_date.",
    ),
    "holdings": EndpointTimeContract(
        endpoint="holdings",
        effective_at_policy="endpoint_as_of_or_observed_date",
        source_as_of_description="Top-level holdings as_of_date supplied by the source payload.",
        effective_at_description=(
            "Use holdings.as_of_date when present, otherwise the observation date."
        ),
    ),
    "ratios": EndpointTimeContract(
        endpoint="ratios",
        effective_at_policy="endpoint_as_of_or_observed_date",
        source_as_of_description="Top-level ratios as_of_date supplied by the source payload.",
        effective_at_description=(
            "Use ratios.as_of_date when present, otherwise the observation date."
        ),
    ),
    "lipper_ratings": EndpointTimeContract(
        endpoint="lipper_ratings",
        effective_at_policy="endpoint_as_of_or_observed_date",
        source_as_of_description=(
            "Universe rows preserve universe_as_of_date; the snapshot does not rely on "
            "ratios.as_of_date."
        ),
        effective_at_description=(
            "Use a top-level endpoint as_of_date when present, otherwise the "
            "observation date."
        ),
        notes="Universe-level as_of_date remains preserved in child rows.",
    ),
    "dividends": EndpointTimeContract(
        endpoint="dividends",
        effective_at_policy="endpoint_as_of_or_observed_date",
        source_as_of_description=(
            "Dividend event rows preserve their own event dates; snapshot metrics use a "
            "top-level endpoint as_of_date when present."
        ),
        effective_at_description=(
            "Use endpoint as_of_date when present, otherwise the observation date."
        ),
    ),
    "morningstar": EndpointTimeContract(
        endpoint="morningstar",
        effective_at_policy="endpoint_as_of_or_observed_date",
        source_as_of_description="Top-level Morningstar as_of_date supplied by the payload.",
        effective_at_description=(
            "Use morningstar.as_of_date when present, otherwise the observation date."
        ),
    ),
    "price_chart": EndpointTimeContract(
        endpoint="price_chart",
        effective_at_policy="series_max_point_or_observed_date",
        source_as_of_description="Price series points preserve their own trade dates.",
        effective_at_description=(
            "Use the latest trade_date present in the payload, otherwise the observation date."
        ),
    ),
    "sentiment_search": EndpointTimeContract(
        endpoint="sentiment_search",
        effective_at_policy="series_max_point_or_observed_date",
        source_as_of_description="Sentiment series points preserve their own trade dates.",
        effective_at_description=(
            "Use the latest sentiment point date present in the payload, otherwise the observation date."
        ),
    ),
    "ownership": EndpointTimeContract(
        endpoint="ownership",
        effective_at_policy="endpoint_as_of_or_observed_date",
        source_as_of_description=(
            "Ownership trade-log rows preserve their own trade dates; snapshot aggregates "
            "fall back to the observation date unless the payload exposes a top-level as_of_date."
        ),
        effective_at_description=(
            "Use endpoint as_of_date when present, otherwise the observation date."
        ),
    ),
    "esg": EndpointTimeContract(
        endpoint="esg",
        effective_at_policy="endpoint_as_of_or_observed_date",
        source_as_of_description="Top-level ESG as_of_date supplied by the payload.",
        effective_at_description=(
            "Use esg.as_of_date when present, otherwise the observation date."
        ),
    ),
}
