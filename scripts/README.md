# Scripts Index and Maintenance Policy

This folder contains runnable utilities grouped by purpose.

## Primary entrypoints (actively maintained)

- run_composer_shared.py
  - Preferred TOML runner when shared initial population parity matters.
- run_toml_statistical_parity.py
  - Preferred parity statistics runner from TOML.
- process_parity_results.py
  - Preferred artifact-level postprocessing (including distance-to-optimum tests).
- generate_experiment_artifacts.py
  - Standard plots/tables from experiment result folders.

## Diagnostic and analysis tools

- diagnose_bias.py
- plot_diagnostics_bias.py
- run_offspring_sweep.py
- run_programmatic_parity_sweep.py
- verify_single_seed_pairing.py

## Utility/reporting tools

- export_pivot_csv.py
- rank_median_summaries.py
- generate_param_grid_toml.py
- simplega_grid_search.py

## Legacy compatibility wrappers

These remain callable for old commands but forward to archived implementations:

- analyze_parity_suite.py
- parity_significance_test.py
- run_parity_sweep.py

Archived implementations live in scripts/_archive/.

## Shell helpers

- run_toy_100seeds.sh
- run_toy_100seeds_sphere_d5.sh

## Policy for future additions

1. Prefer adding options to an existing primary entrypoint before creating a new script.
2. If a new script is required, document it here under the correct section.
3. If a script becomes superseded, move implementation to scripts/_archive and leave a thin forwarding wrapper.
4. Keep Make targets mapped only to primary entrypoints unless there is a clear historical compatibility reason.
