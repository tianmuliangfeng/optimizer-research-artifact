# Local mechanism closure package

This script freezes the accepted mechanism evidence after MDP-04, MDP-05, GEO-01A, and GEO-01B. It performs no training and requires no GPU.

The build deliberately separates three statements:

1. scheduled down-projection refresh has a replicated short-horizon loss effect;
2. the immediate counterfactual line loss is accurately closed by a local Taylor expansion;
3. neither MDP-05 nor GEO-01B establishes an origin-independent scalar predictor of the short-horizon harm.

`closure_contract.json` pins all local inputs and the GEO-01B ZIP. `build_mechanism_closure.py` verifies hashes, validates every file listed by the GEO-01B handoff manifest, independently recomputes the GEO-01B summaries, and writes immutable tables, figures, snapshots, and a report. The workbook is a presentation layer built from those frozen CSV files and then added to the final manifest.

Typical local commands from the repository root:

```powershell
python scripts\mechanism_closure\build_mechanism_closure.py check
python scripts\mechanism_closure\build_mechanism_closure.py build --output-dir runs\_shared\analysis\mechanism_closure_20260805
# Build the workbook with a Node environment that provides @oai/artifact-tool.
node scripts\mechanism_closure\build_workbook.mjs <package-dir> <preview-dir>
python scripts\mechanism_closure\build_mechanism_closure.py finalize --output-dir runs\_shared\analysis\mechanism_closure_20260805
python scripts\mechanism_closure\validate_mechanism_closure.py --output-dir runs\_shared\analysis\mechanism_closure_20260805
```

The final package is immutable. A rebuild must use a new output directory.
