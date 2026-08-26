# Testing and coverage

The repository keeps correctness and coverage as separate CI signals. The normal
Rust job stays on stable, while the Rust coverage job alone uses nightly for LLVM
branch instrumentation. Rust, Python, and browser JavaScript each have an
independent **80% production-line coverage** gate. Branch coverage is collected
and shown, but it is not a blocking metric.

## Coverage scope

| Runtime | Included production code | Gate |
| --- | --- | --- |
| Rust | `voxel-core`, `voxel-view`, and `voxel-py` | Workspace lines ≥80% |
| Python | `python/voxelgym` and `web/server.py` | Aggregate lines ≥80% |
| Browser JavaScript | `web/static/app.js` | Lines ≥80% |

Tests, `bench/`, diagnostic `scripts/`, generated files, HTML, and CSS are not
part of these denominators. Rust inline test modules are disabled for coverage so
test implementation lines cannot inflate the result. The Rust report includes a
separate text view for every crate, but only the workspace line total is gated.
CI also verifies that `crates/voxel-py/src/lib.rs` has executed lines; merely
listing the PyO3 crate in a report is not enough.

## Local setup

Python 3.11, Rust stable, and Node.js 22 match CI. Create and activate a virtual
environment before using `maturin develop`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -e "python[dev]"
maturin develop --release -m crates/voxel-py/Cargo.toml

Set-Location web
npm ci
Set-Location ..
```

Run the ordinary correctness suites with stable Rust and the complete Python test
set, including tests marked `ml`:

```powershell
cargo +stable test --workspace --release
python -m pytest python/tests -q
npm --prefix web test
```

## Python coverage

The root `.coveragerc` enables branch collection, relative paths, and missing-line
output. Run the full suite and generate HTML, XML, JSON, the raw coverage database,
and a terminal summary:

```powershell
New-Item -ItemType Directory -Force coverage/python | Out-Null
$env:COVERAGE_FILE = "coverage/python/.coverage"
python -m pytest python/tests -q `
  --cov=python/voxelgym --cov=web --cov-config=.coveragerc `
  --cov-report=term-missing `
  --cov-report=html:coverage/python/html `
  --cov-report=xml:coverage/python/coverage.xml `
  --cov-report=json:coverage/python/coverage.json `
  --cov-fail-under=0

$report = Get-Content -Raw coverage/python/coverage.json | ConvertFrom-Json
$lineRate = 100 * $report.totals.covered_lines / $report.totals.num_statements
if ($lineRate -lt 80) { throw "Python line coverage is $lineRate%, below 80%" }
```

`--cov-fail-under=0` is intentional in this command: with branch measurement
enabled, coverage.py's combined percentage includes branch opportunities. The
separate line calculation enforces the project's line-only contract while the
same report still exposes branch coverage.

## Browser JavaScript coverage

Vitest loads the real page under jsdom and executes the production self-starting
script with browser fakes. V8 writes JSON and HTML reports and enforces the line
threshold configured in `web/vitest.config.js`:

```powershell
npm --prefix web run coverage
```

The local report is under `web/coverage/`.

## Rust coverage

The merged Rust report follows cargo-llvm-cov's external-tests flow. It must use
one shared LLVM instrumentation environment for Cargo tests, the Maturin build,
and non-ML pytest execution. Run this sequence in Bash, Git Bash, or WSL:

```bash
rustup toolchain install nightly --profile minimal --component llvm-tools-preview
cargo install cargo-llvm-cov --locked
python -m pip install -e "python[dev]"

mkdir -p coverage/rust target/wheels
export RUSTUP_TOOLCHAIN=nightly
eval "$(cargo llvm-cov show-env --sh --branch)"
cargo llvm-cov clean --workspace

cargo test --workspace --release
maturin build --release -m crates/voxel-py/Cargo.toml --out target/wheels
python -m pip install --force-reinstall --no-deps target/wheels/*.whl
python -m pytest python/tests -q -m "not ml"

cargo llvm-cov report --release --branch --html --output-dir coverage/rust
cargo llvm-cov report --release --branch --json --output-path coverage/rust/coverage.json
cargo llvm-cov report --release --branch --lcov --output-path coverage/rust/lcov.info
cargo llvm-cov report --release --branch > coverage/rust/summary.txt
cargo llvm-cov report --release --branch --fail-under-lines 80
```

The VQA model coverage module is marked `ml` as a whole and skips cleanly during
collection when PyTorch is absent. All of its tests run in the complete Python
coverage job; every non-ML pytest test still executes against the instrumented
native extension.

## CI artifacts

Coverage jobs append their text report to the GitHub Actions job summary and use
`if: always()` for artifact upload, including when a threshold fails:

- `rust-coverage`: HTML, LLVM JSON, LCOV, workspace text, and per-crate text.
- `python-coverage`: HTML, XML, JSON, raw `.coverage`, and terminal text.
- `web-coverage`: V8 HTML, JSON, and terminal text.

No coverage data is sent to an external coverage service.
