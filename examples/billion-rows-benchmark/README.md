# Billion-rows benchmark (example)

This example contains a benchmark harness that runs a pipeline against a static
CSV file (one integer per line). It's intended as a reproducible example for
large-file benchmarks such as the "billion rows" challenge.

Files
- `benchmark_test.go`: the benchmark. It expects a static fixture file with
  one integer per line and emits those values through a simple CSV source.
- `fixtures/sample.csv`: a tiny sample fixture included for CI and quick runs.
- `scripts/expand_fixture.sh`: simple shell script to expand the small sample
  into a larger fixture by repeating lines.

How to run
1. Use the small sample (fast / CI):

```bash
# from repo root
go test -run=^$ -bench BenchmarkMillionRowChallenge ./examples/billion-rows-benchmark -benchmem
```

2. Use a larger static fixture (recommended for real measurements):

- Obtain or generate a static CSV where each line is an integer (the 1B
  challenge provides generators). Place it at `examples/billion-rows-benchmark/fixtures/1brc.csv` or set `FIXTURE_PATH`.

Example to expand the included sample to 1_000_000 lines (quick, not realistic):

```bash
cd examples/billion-rows-benchmark
mkdir -p fixtures
./scripts/expand_fixture.sh fixtures/sample.csv fixtures/1m.csv 1000000
export FIXTURE_PATH=$(pwd)/fixtures/1m.csv
# run the bench (this will still run the benchmark harness, which runs the pipeline once per iteration)
go test -run=^$ -bench BenchmarkMillionRowChallenge -benchmem
```

Notes
- The benchmark accepts `FIXTURE_PATH` environment variable to point to the CSV fixture. If not set, it falls back to `fixtures/sample.csv` included in the example.
- For the official 1B challenge, follow the instructions in the challenge repository to generate the required static file and set `FIXTURE_PATH` to that file.
- The benchmark uses the per-subscription helper `ro.WithObserverPanicCaptureDisabled(ctx)` to avoid mutating global state when measuring hot-path performance.
