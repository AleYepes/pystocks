from __future__ import annotations

from collections.abc import Iterable

import pycountry

RISK_FREE_SERIES_BY_ECONOMY: dict[str, str] = {
    "USA": "DTB3",
    "CAN": "IR3TIB01CAM156N",
    "DEU": "IR3TIB01DEM156N",
    "GBR": "IR3TIB01GBM156N",
    "FRA": "IR3TIB01FRA156N",
}

WORLD_BANK_INDICATOR_MAP: dict[str, str] = {
    "SP.POP.TOTL": "population",
    "NY.GDP.PCAP.CD": "gdp_pcap",
    "NY.GDP.MKTP.CD": "economic_output_gdp",
    "BX.KLT.DINV.WD.GD.ZS": "foreign_direct_investment",
    "NE.IMP.GNFS.ZS": "imports_goods_services",
    "NE.EXP.GNFS.ZS": "exports_goods_services",
}

_INVALID_ECONOMY_CODES: frozenset[str] = frozenset(
    {
        "",
        "Unidentified",
    }
)


def normalize_economy_code(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    upper = text.upper()
    if upper in _INVALID_ECONOMY_CODES:
        return None
    if len(upper) == 3 and upper.isalpha():
        country = pycountry.countries.get(alpha_3=upper)
        return str(country.alpha_3) if country is not None else None
    if len(upper) == 2 and upper.isalpha():
        country = pycountry.countries.get(alpha_2=upper)
        return str(country.alpha_3) if country is not None else None
    country = pycountry.countries.get(name=text)
    if country is not None:
        return str(country.alpha_3)
    try:
        matches = pycountry.countries.search_fuzzy(text)
    except LookupError:
        return None
    return str(matches[0].alpha_3) if matches else None


def normalize_economy_codes(values: Iterable[object]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        code = normalize_economy_code(value)
        if code is None or code in seen:
            continue
        normalized.append(code)
        seen.add(code)
    return normalized
