# Shipping Data Pipeline

Tools for extracting, cleaning, and temporally aligning FactSet shipping data
so it can be compared directly with the existing supply-chain datasets.

## Structure

- `01_extract_transactions.py` – stream raw shipment transactions from FactSet SQL
  up to a specified cutoff date into a single parquet file. By default it
  includes both `record_status = 'N'` ("new") and `record_status = 'A'`
  ("amended") records and can optionally apply a minimum `record_date`
  to bound the extract window. Supports choosing the train, validation, or
  full dataset boundary.
- `02_clean_transactions.py` – deduplicate and sanitize the raw feed, removing
  placeholder entities and ensuring we only retain factual shipper->consignee
  relationships.
- `03_split_transactions.py` – mirror the core pipeline splits (train/val/test)
  using the same temporal boundaries so evaluation remains apples-to-apples.
- `common.py` – helpers for loading temporal metadata and computing shared
  date/epoch conversions.

All scripts expose a CLI (`python -m data_processing.shipping.<script>`)
with sensible defaults and can also be imported as modules.
