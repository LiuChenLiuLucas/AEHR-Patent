# Reproduction protocol

The primary dataset is the remote `uspto-release` directory. The sibling
`uspto-debug` directory is reserved for smoke runs only. Before any full run,
execute `tools/audit_dataset.py` on both directories and retain the JSON
reports with the run artifacts.

The primary research target is an aspirational absolute +0.10 improvement in
full-test hybrid Macro-F1 over the paper's 5% hybrid reference (0.412 to
0.512). This target is not a guaranteed result. Micro-F1, hierarchical F1,
rare-label F1, route fraction, runtime, and GPU cost must be reported with it.

The original upstream files are listed in `UPSTREAM_SHA256SUMS.txt`. Local
changes are intentionally kept in separate files or explicitly marked in the
modified scripts so that the reproduction baseline remains auditable.

Before training, run the standard-library audit against both data directories:

```bash
python tools/audit_dataset.py /path/to/uspto-debug --output runs/audit_debug.json
python tools/audit_dataset.py /path/to/uspto-release --output runs/audit_release.json
```

The planned 14-day schedule starts on 2026-10-13. Full training is gated on
the independent environment, validated data manifest, model-load smoke test,
and confirmation that GPU 3 is available without terminating another user's
process.
