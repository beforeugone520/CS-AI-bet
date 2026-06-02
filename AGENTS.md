# Project Agent Instructions

## Python ML Acceleration

This project supports optional accelerated model backends. Prefer these libraries for training and prediction speed when they are installed:

- `numpy`
- `scikit-learn`
- `xgboost`
- `joblib`

Use the PyPI package name `scikit-learn`; do not install the deprecated `sklearn` package.

The default model stack should use:

- `sklearn.linear_model.LogisticRegression` for the logistic component.
- `sklearn.ensemble.RandomForestClassifier` for the random forest component.
- `xgboost.XGBClassifier` for the xgboost component.
- `joblib` to parallelize independent base-model fitting when practical.

Keep the pure-Python model implementations as a reliable fallback. If an accelerated dependency is missing or fails to import, the code should continue running with the pure-Python backend instead of crashing.

The neural-network component should remain on the existing pure-Python implementation by default. The sklearn MLP path may be enabled only when explicitly requested with `CS2PICKEM_ACCELERATED_MLP=1`, because tiny sample sets can produce noisy numerical warnings.

## Local Python Environment

For this workspace, the acceleration packages were installed into the user Python environment with:

```bash
python3 -m pip install --user --upgrade numpy scikit-learn xgboost joblib
```

On macOS, the installed `xgboost` wheel may require `libomp.dylib`. In this workspace, `libxgboost.dylib` was patched to use the `libomp.dylib` bundled with scikit-learn:

```bash
install_name_tool -change @rpath/libomp.dylib \
  /Users/bruce/Library/Python/3.9/lib/python/site-packages/sklearn/.dylibs/libomp.dylib \
  /Users/bruce/Library/Python/3.9/lib/python/site-packages/xgboost/lib/libxgboost.dylib
```

Before relying on accelerated training, verify the active backends:

```bash
PYTHONPATH=src python3 - <<'PY'
from cs2pickem.models import default_ensemble
print(default_ensemble(seed=7, epochs=2, n_jobs=1).component_backends)
PY
```

Expected default output after acceleration is available:

```text
{'logistic': 'sklearn', 'random_forest': 'sklearn', 'xgboost': 'xgboost', 'neural_network': 'pure_python'}
```

## Verification

After changing model code or dependency behavior, run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

For a quick runtime smoke test, run:

```bash
PYTHONPATH=src python3 -m cs2pickem.cli demo
```
