# Pystocks `/src` Gap Inventory

This document compares the current `/pystocks` capability inventory against the older `/src` implementation and notebooks.

It focuses on capabilities and workflows, not one-to-one file migration.

The goal is to identify:

- capabilities already preserved in `/pystocks`
- capabilities that appear weakened or missing
- capabilities that may be legacy noise and not worth reviving

## Method

This comparison is based on:

- [src/analysis.py](/Users/alex/Documents/pystocks/src/analysis.py)
- [src/functions.py](/Users/alex/Documents/pystocks/src/functions.py)
- notebook structure and code-cell headings in:
  - [src/1.api_contracts.ipynb](/Users/alex/Documents/pystocks/src/1.api_contracts.ipynb)
  - [src/2.ibkr_ocr.ipynb](/Users/alex/Documents/pystocks/src/2.ibkr_ocr.ipynb)
  - [src/3.data_prep.ipynb](/Users/alex/Documents/pystocks/src/3.data_prep.ipynb)
  - [src/4.justetf.ipynb](/Users/alex/Documents/pystocks/src/4.justetf.ipynb)
  - [src/5.api_series.ipynb](/Users/alex/Documents/pystocks/src/5.api_series.ipynb)
  - [src/6.analysis.ipynb](/Users/alex/Documents/pystocks/src/6.analysis.ipynb)

Some notebook-derived conclusions are necessarily approximate, but the high-level capability signals are clear enough to be useful.

## Capability Families Already Preserved In `/pystocks`

These legacy concerns appear to have a clear modern equivalent:

- product-universe bootstrap from IBKR catalog
- repeated fundamentals collection across a maintained universe
- raw plus canonical storage for fundamentals endpoints
- cleaned price history and return construction
- supplementary World Bank and risk-free data handling
- point-in-time panel construction
- factor construction, screening, and regression-based research

These are not identical implementations, but the capability families are present.

## Capabilities Present In `/src` That Look Weakened Or Missing

### 1. Portfolio construction and optimization

This is the biggest functional gap.

Legacy evidence:

- `src/6.analysis.ipynb` includes optimization functions such as:
  - `optimize_convex`
  - `optimize_non_convex`
  - `reduce_inputs`
  - `print_portfolio_stats`
  - factor-constrained portfolio evaluation loops

Current `/pystocks` state:

- current analysis produces factors, expected returns, and betas
- current runtime surface does not appear to expose actual portfolio optimization or efficient-frontier construction as a first-class capability

Assessment:

- this looks like a real missing top-level capability, not mere implementation noise

### 2. Universe curation and tradability verification

Legacy evidence:

- `src/analysis.py:verify_files()` checks IB contract details and rejects mismatches
- `src/functions.py` contains:
  - `sort_by_eur_exchanges()`
  - leveraged/multiplier filtering helpers
  - row-validation and cleaning logic around scraped contracts
- notebook `src/1.api_contracts.ipynb` includes verification and save steps

Current `/pystocks` state:

- `/pystocks` bootstraps the product universe and stores metadata
- it does not appear to preserve the richer legacy workflow around tradability verification, exchange prioritization, and leveraged-product exclusion as a first-class capability

Assessment:

- likely worth reconsidering in the rebuild as part of explicit universe-governance requirements

### 3. JustETF enrichment

Legacy evidence:

- `src/4.justetf.ipynb` includes `scrape_etf_profile(isin)` and related scraping flow
- notebook comments suggest use of fields such as distribution policy and replication

Current `/pystocks` state:

- no obvious current equivalent in the runtime CLI or storage schema

Assessment:

- likely a real missing external enrichment capability
- whether it belongs in the rebuild depends on how valuable those fields are for universe curation or analysis

### 4. Historical-series collection from direct trading APIs

Legacy evidence:

- `src/5.api_series.ipynb` contains:
  - `get_historical(...)`
  - `save_data(...)`
  - `fill_internal_gaps(...)`
- legacy workflow appears to fetch and maintain local historical series files directly

Current `/pystocks` state:

- `/pystocks` stores price series via the IBKR fundamentals/performance payload path
- there is no obvious current equivalent for a distinct “direct historical API series collection” capability

Assessment:

- may be partially replaced by the current fundamentals/price-chart ingestion path
- still worth recording as a potentially missing fallback or validation path

### 5. Richer raw-contract cleaning and explosion workflow

Legacy evidence:

- `src/3.data_prep.ipynb` includes:
  - file loading
  - exploding nested raw columns
  - correcting profile net assets and TER
  - class refinement
  - missingness filtering and imputation
  - merging top-10 columns

Current `/pystocks` state:

- `/pystocks` moved toward endpoint-specific canonical tables and snapshot feature preprocessing
- there is no direct analog to the old “explode one giant raw contract table into research-ready columns” workflow

Assessment:

- much of this may be intentionally superseded by the canonical storage model
- some pieces may still matter, especially:
  - classification refinement
  - specific domain cleaning rules
  - net-assets normalization logic

### 6. Explicit file-level/manual review workflows

Legacy evidence:

- several notebooks include manual review, duplicate checks, exact value checks, and manual save steps
- the OCR notebook is especially operational and semi-manual

Current `/pystocks` state:

- diagnostics exist, but the old manual-review workflow is not represented as an explicit operational capability

Assessment:

- probably not a core rebuild target as-is
- but some form of review/audit workflow may still be valuable

## Capabilities In `/src` That Likely Should Not Be Carried Forward Directly

### OCR-driven TWS scraping

Legacy evidence:

- [src/2.ibkr_ocr.ipynb](/Users/alex/Documents/pystocks/src/2.ibkr_ocr.ipynb)

Assessment:

- this looks like historical acquisition scaffolding rather than a desirable target architecture concern
- preserve only if a fallback acquisition mode is truly needed

### Giant flat research table shaping

Legacy evidence:

- `src/3.data_prep.ipynb` appears to rely heavily on exploding and cleaning one large contract dataframe

Assessment:

- this is exactly the shape the newer canonical storage model was trying to move away from
- do not revive the flat-table architecture just to recover a few useful transformations

## `/src` Features Worth Evaluating For Explicit Recovery

These are the strongest candidates for explicit consideration in the rebuild requirements:

1. Portfolio optimization and efficient-frontier construction
2. Universe verification and exclusion policy
3. JustETF enrichment
4. Direct historical-series fallback or validation path
5. Selected domain-cleaning rules from the old prep notebook

## Recommended Next Use Of This Comparison

When revising the FRD, explicitly decide for each of the five items above:

- `required`
- `optional`
- `legacy / do not carry forward`

That will prevent the rebuild from accidentally omitting portfolio construction while also preventing accidental reintroduction of notebook-era operational clutter.
