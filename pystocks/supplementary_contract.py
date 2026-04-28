from typing import Final

import pycountry

WORLD_BANK_INDICATOR_MAP: Final[dict[str, str]] = {
    "SP.POP.TOTL": "population",
    "NY.GDP.PCAP.CD": "gdp_pcap",
    "NY.GDP.MKTP.CD": "economic_output_gdp",
    "BX.KLT.DINV.WD.GD.ZS": "foreign_direct_investment",
    "NE.IMP.GNFS.ZS": "imports_goods_services",
    "NE.EXP.GNFS.ZS": "exports_goods_services",
}

RISK_FREE_SERIES_BY_ECONOMY: Final[dict[str, str]] = {
    "USA": "DTB3",
    "CAN": "IR3TIB01CAM156N",
    "DEU": "IR3TIB01DEM156N",
    "GBR": "IR3TIB01GBM156N",
    "FRA": "IR3TIB01FRA156N",
}

SUPPLEMENTARY_TIME_CONTRACT: Final[dict[str, dict[str, str]]] = {
    "supplementary_risk_free_sources": {
        "observed_at": "fetched_at",
        "source_as_of": "trade_date",
        "effective_at": "trade_date",
    },
    "supplementary_world_bank_raw": {
        "observed_at": "fetched_at",
        "source_as_of": "year",
        "effective_at": "year-end date chosen during preprocess",
    },
    "supplementary_risk_free_daily": {
        "observed_at": "max fetched_at across contributing series rows",
        "source_as_of": "trade_date",
        "effective_at": "trade_date",
    },
    "supplementary_world_bank_country_features": {
        "observed_at": "max fetched_at across contributing raw rows",
        "source_as_of": "feature_year",
        "effective_at": "December 31 of feature_year",
    },
}


def normalize_economy_code(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    upper = text.upper()
    if len(upper) == 3 and upper.isalpha():
        country = pycountry.countries.get(alpha_3=upper)
        return country.alpha_3 if country else upper
    if len(upper) == 2 and upper.isalpha():
        country = pycountry.countries.get(alpha_2=upper)
        return country.alpha_3 if country else upper
    country = pycountry.countries.get(name=text)
    if country:
        return country.alpha_3
    try:
        matches = pycountry.countries.search_fuzzy(text)
    except LookupError:
        return upper
    return matches[0].alpha_3 if matches else upper
