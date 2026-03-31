# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`samber/ro` is a Go implementation of the ReactiveX spec — a library for reactive/stream programming with Observables, Observers, and Operators. It uses Go 1.18+ generics extensively. The library is v0 and follows SemVer strictly.

## Build & Test Commands

```bash
make build                    # Build all modules
make test                     # Run all tests with race detector
make lint                     # Run golangci-lint + license header check
make lint-fix                 # Auto-fix lint issues
make bench                    # Run benchmarks
make coverage                 # Generate coverage report (cover.html)
```

Run a single test:
```bash
go test -race -run TestFunctionName ./...
```

Run tests for a specific plugin:
```bash
cd plugins/encoding/json && go test -race ./...
```

## Multi-Module Architecture

This is a **Go workspace** (`go.work`) with many independent modules. The root module is `github.com/samber/ro`. Each plugin under `plugins/` has its own `go.mod`. Some plugins are commented out in `go.work` because they require newer Go versions.

The SIMD plugin (`plugins/exp/simd`) requires `GOEXPERIMENT=simd` and `GOWORK=off` to build/test.

## Code Layout

- **Root package (`ro`)** — Core types and all built-in operators
- **`internal/`** — Internal helpers: `xsync` (mutex wrappers), `xatomic`, `xrand`, `xtime`, `xerrors`, `constraints`
- **`testing/`** — Package `rotesting` with `AssertSpec[T]` interface for fluent test assertions
- **`plugins/`** — Each plugin is a separate Go module with its own `go.mod`. Categories: encoding, observability, rate limiting, I/O, data manipulation, etc.
- **`ee/`** — Enterprise Edition (separate license). Contains `otel` and `prometheus` plugins, plus licensing infrastructure. See [`ee/README.md`](ee/README.md) and the [`ee/cmd/license` CLI](ee/cmd/license/README.md)
- **`examples/`** — Working example applications (each is its own module)
- **`docs/`** — Docusaurus documentation site. Has its own [`CLAUDE.md`](docs/CLAUDE.md) for doc-writing conventions

## Operator Pattern

All chainable operators follow this signature pattern:
```go
func OperatorName[T any](params) func(Observable[T]) Observable[R]
```

Example:

```go
func Filter[T any](predicate func(item T) bool) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						ok := predicate(value)
						if ok {
							destination.NextWithContext(ctx, value)
						}
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}
```

They return a function that transforms one Observable into another, enabling composition via `Pipe()`.

### Operator Variant Suffixes

Most operators have variants created by combining these suffixes:

- **`I`** — Adds an `index int64` parameter to the callback (e.g., `FilterI`, `MapI`)
- **`WithContext`** — Adds `context.Context` to the callback signature (e.g., `FilterWithContext`, `MapWithContext`)
- **`Err`** — The callback can return an `error` that terminates the stream (e.g., `MapErr`)

These suffixes combine in a fixed order: **`Err` + `I` + `WithContext`**. Examples:
- `Map` → `MapI` → `MapWithContext` → `MapIWithContext`
- `MapErr` → `MapErrI` → `MapErrWithContext` → `MapErrIWithContext`

Other naming patterns:
- **Numbered suffixes** (2, 3, 4, 5...) — Arity variants for multi-observable operators (e.g., `CombineLatest2`, `Zip3`, `MergeWith1`). Also used for type-safe pipe: `Pipe1` through `Pipe11`
- **`Op`** — Operator version of a creation function, for use inside `Pipe()` (e.g., `PipeOp`)

## Core Operators vs Plugins

**Core operators** live in the root `ro` package and have no external dependencies beyond `samber/lo`. They cover the standard ReactiveX operator categories (creation, transformation, filtering, combining, etc.) and are imported as `github.com/samber/ro`.

**Plugins** are separate Go modules under `plugins/`, each with its own `go.mod` and third-party dependencies. They follow the same `func(Observable[T]) Observable[R]` signature pattern and compose with core operators via `Pipe()`. Plugins wrap external libraries (e.g., `zap`, `sentry`, `fsnotify`) or provide domain-specific operators (e.g., JSON encoding, CSV I/O, rate limiting). Import them separately, e.g., `github.com/samber/ro/plugins/encoding/json`.

## Testing Conventions

- Tests use `testify` assertions and `go.uber.org/goleak` for goroutine leak detection
- Test files follow Go convention: `foo_test.go` alongside `foo.go`
- Example tests use `_example_test.go` suffix
- The `plugins/testify` plugin provides reactive stream assertion helpers

Typical test pattern — use `Collect()` to gather all emitted values and assert:

```go
func TestOperatorTransformationMap(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		Map(func(v int) int { return v * 2 })(Just(1, 2, 3)),
	)
	is.Equal([]int{2, 4, 6}, values)
	is.NoError(err)

	// Test error propagation
	values, err = Collect(
		Map(func(v int) int { return v * 2 })(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}
```

Always test edge cases with `Empty[T]()` (empty source) and `Throw[T](assert.AnError)` (error source). Also test early unsubscription, context propagation, and context cancellation where applicable.

## Contributing Conventions

Full guides: [`docs/docs/contributing.md`](docs/docs/contributing.md) (contributing) and [`docs/docs/hacking.md`](docs/docs/hacking.md) (writing custom operators/plugins).

- **Operator naming**: Must be self-explanatory and respect ReactiveX/RxJS standards. Inspired by https://reactivex.io/documentation/operators.html and https://rxjs.dev/api
- **Context propagation**: Operators must not break the context chain. Always use `SubscribeWithContext`, `NextWithContext`, `ErrorWithContext`, `CompleteWithContext`. The `WithContext` variant callbacks receive and return a `context.Context`
- **Callback naming**: `predicate` for bool-returning callbacks, `transform`/`project` for value-transforming callbacks, `callback` for void callbacks
- **Variadic operators**: Some operators accept `...Observable[T]` for flexibility (e.g., `Zip`, `Merge`, `MergeWith`)
- **Type aliases**: Some operators use `~[]T` constraints to accept named slice types, not just `[]T` (e.g., `Flatten`)
- **Documentation**: Each operator needs a Go Playground link in its comment, a markdown doc in [`docs/data/`](docs/data/) (one file per operator, e.g. `core-map.md`, `plugin-encoding-json-marshal.md`), an example in `ro_example_test.go`, and an entry in `docs/static/llms.txt`. See [`docs/CLAUDE.md`](docs/CLAUDE.md) for the full doc-file format
- **License headers**: All `.go` files require license headers (`licenses/header.apache.txt` for open source, `licenses/header.ee.txt` for `ee/`). Run `make lint` to verify. Full license texts: [`licenses/LICENSE.apache.md`](licenses/LICENSE.apache.md) (Apache 2.0, open-source code) and [`licenses/LICENSE.ee.md`](licenses/LICENSE.ee.md) (EE code under `ee/`)
- **Update the documentation**: when updating a feature of the project, you MUST update the documentation. See @./docs/CLAUDE.md

## References

- **Contribution guidelines**: @./docs/docs/contributing.md
- **Extending ro guidelines**: @./docs/docs/hacking.md
- **Documentation guidelines**: @./docs/CLAUDE.md
- **Troubleshooting guidelines**: @./docs/docs/troubleshooting/
- If you need more context on the project, read the **LLMs documentation**: @./docs/static/llms.txt
