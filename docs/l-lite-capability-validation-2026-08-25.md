# L-lite capability validation — 2026-08-25

## Scope

This validation covers rule-based legality, compiler materialization,
final-artifact observation and the standalone native-autotune control API.  It
contains no performance measurement and grants no speedup, model-fit or
release authority.

The repository is based on Triton 3.6.0.  The Loop plan provenance contract
pins compiler commit `7c56a5e40f7fd928dfd5c72902d5def0097db73a`.

## Capability identity

`lib/Dialect/Triton/Transforms/HBVLoop.cpp` has SHA-256:

```text
adfdb0ac5ff0ce9ec301ec5349d8be9e801fcfd0fad3340fb46de87c7feb6b31
```

At validation time this byte-identically matched the compiler-side capability
implementation used by the main research branch.  L-lite does not import that
branch's predictor, fitted model, residual, uncertainty or release selector.

## Build environment and result

- build type: `TritonRelBuildWithAsserts`;
- codegen backends required by the Triton Python extension: `nvidia;amd`;
- Proton profiler: disabled;
- maximum parallel compilation processes: four;
- CUDA assembler: 12.8.93;
- built targets: `libtriton.so` and `triton-opt`;
- result: passed.

The public source checkout requires Triton's standard editable-install backend
links, or equivalent links from `python/triton/backends/{nvidia,amd}` to the
in-tree backend packages.  CUDA tool binaries are external build/runtime
dependencies and are not committed.

## Tests

The standalone Python control surface imported from this repository and
passed:

```text
python/test/unit/l_lite/test_control.py: 4 passed
```

All public Loop compiler tests were run through the independently built
`triton-opt`, with one test worker:

```text
test/Triton/hbv-loop-*.mlir: 13 passed
```

The compiler tests cover positive and negative Bridge footprint intervals,
operation-neutral Bridge handling, pure-call closure, recursive effect
containers, ungrouped-axis translation, mixed-radix independence, runtime-mask
authority, early-return normalization and real Bridge materialization.

## Clean-publication audit

The committed tree contains no private iteration directory, series plan or
memory, machine-local measurement, workload-specific adapter, backend
predictor, fitted coefficient, residual, uncertainty model or generated
binary.  Build directories, backend symlinks, Python caches and shared objects
remain untracked.
