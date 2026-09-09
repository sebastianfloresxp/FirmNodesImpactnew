# db_client

Ad-hoc utilities for querying the FactSet SQL Server without touching the main
pipeline code.

## Environment

Populate the following variables (a `.env` file in the repo root is supported):

- `DB_SERVER`
- `DB_DATABASE`
- `DB_USERNAME`
- `DB_PASSWORD`

## Quick start

Run a query directly from the command line:

```bash
python -m db_client.cli --sql "SELECT TOP (10) * FROM sym_v1.sym_entity"
```

Read a query from a file and stream to CSV:

```bash
python -m db_client.cli --sql-file queries/sample.sql --chunksize 10000 --output results/sample.csv
```

Inspect the resolved SQL without running it:

```bash
python -m db_client.cli --sql-file queries/sample.sql --param as_of='2024-01-01' --dry-run
```

## Supply chain export

Create a parquet edge list plus relationship histogram for the FactSet
`ent_scr_relationships` feed:

```bash
python -m db_client.export_supply_chain_edges \
  --edges-out data/supply_edges.parquet \
  --histogram-out data/supply_histogram.csv \
  --rel-type SUPPLIER --rel-type CUSTOMER \
  --as-of-date 2024-12-31
```

Available filters:

- `--rel-type`: repeatable filter to limit the relationship types
- `--start-date-min` / `--start-date-max`: restrict by relationship start date
- `--as-of-date`: keep only links active on the given date

## Programmatic usage

```python
from db_client import run_query

df = run_query("SELECT TOP (5) factset_entity_id FROM sym_v1.sym_entity")
print(df)
```

Chunked iteration:

```python
from db_client import iter_query

for batch in iter_query("SELECT * FROM large_table", chunksize=10000):
    process(batch)
```
