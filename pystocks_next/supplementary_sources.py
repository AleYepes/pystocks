from __future__ import annotations

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
