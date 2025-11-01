# AGENTS

## Getting oriented
- This repo hosts `samber/ro`, a Go implementation of the ReactiveX observable/observer model. All packages rely heavily on Go 1.18+ generics; stick with `go1.20` or newer to run the full module set (`go.work` enumerates optional plugin modules that require newer Go versions).
- Core developer loops:
  - Build/tests: `go test -race ./...` (or `make test`) is the standard check; benchmarks use `make bench`. `make coverage` produces an HTML report. Linting combines `golangci-lint` and a license header check via `make lint`.
  - Project uses a `go.work` workspace that pulls in numerous plugin submodules and examples. When adding a new package, ensure it is listed in `go.work` if it should build as part of the workspace.
- All Go sources start with the standard Apache 2 header (see existing files). Keep that header on new files.

## Architectural primer
- **Notifications** (`ro.go`): Values flow through `Notification[T]` objects tagged with `KindNext`, `KindError`, or `KindComplete`. Global hooks `OnUnhandledError` and `OnDroppedNotification` centralize side effects when observers drop signals or panics occur.
- **Observable graph**:
  - `Observable[T]` (`observable.go`) is the core abstraction. `NewObservable*` helpers capture subscription logic and optionally enforce concurrency via `ConcurrencyMode` and `Backpressure`. The observable factory returns a `Teardown` that cleans up when downstream unsubscribes.
  - `Observer[T]` (`observer.go`) consumes notifications. Concrete observers wrap callbacks and guard them with atomic state transitions and `lo.TryCatchWithErrorValue` to convert panics into `OnUnhandledError` calls.
  - `Subscriber[T]` (`subscriber.go`) pairs an `Observer` with a `Subscription`, mediating concurrency (`BackpressureBlock` vs `BackpressureDrop`) using custom mutexes from `internal/xsync`. Most operators eventually wrap destinations in a subscriber and return `sub.Unsubscribe` as their teardown.
  - `Subscription` (`subscription.go`) aggregates teardown callbacks and guarantees they run once, turning panics into `unsubscriptionError`s via `lo.TryCatchWithErrorValue`.
- **Operators**: Every operator returns a function `func(Observable[A]) Observable[B]`. The pattern is:
  1. Wrap the source with `NewUnsafeObservableWithContext` or another `NewObservableWith*` helper to create a new observable.
  2. Inside, `SubscribeWithContext` to the upstream source using `NewObserverWithContext` for proper context propagation.
  3. Forward notifications to the destination observer, and return `sub.Unsubscribe` as the teardown. Use `recoverUnhandledError` when spawning goroutines so unexpected panics route through `OnUnhandledError`.
  4. Honor concurrency/backpressure semantics of the helper you pick (eg. use `NewObservableWithConcurrencyMode` if you need a specific `ConcurrencyMode`).
- **Composition APIs** (`pipe.go`): `Pipe` chains a source with runtime type-checked operators (reflection). Prefer `Pipe1`…`Pipe10`/`PipeOpN` for compile-time safety.
- **Subjects** (`subject*.go`): Provide bridge types that act as both observer and observable. Special subjects (`Publish`, `Behavior`, `Replay`, `Async`, `Unicast`) add buffering semantics—consult the respective file before modifying to preserve invariants about subscriber counts, replay caches, or concurrency guarantees.
- **Internal toolkits** (`internal/`):
  - `xsync` supplies tailored mutex implementations (eg. spin-free fast mutex, try-lock support) tuned for the subscriber backpressure behavior.
  - `xtime`, `xrand`, `xerrors`, etc., are thin wrappers used throughout operators; prefer them over reimplementing utilities.
- **Plugins & EE**: `plugins/` hosts additional operator sets organized as independent Go modules. Check the specific module’s README/tests before altering shared APIs—they often mirror core operator patterns but may add IO/external dependencies. `ee/` contains enterprise-only extensions; only `ee/plugins/prometheus` is enabled in `go.work` by default.

## Implementation tips
- **Context propagation**: Operator callbacks typically receive `ctx` and must pass the context returned by upstream callbacks into downstream notifications. When altering or introducing operators, ensure contexts are forwarded (see `MapIWithContext` for reference).
- **Error propagation**: Operators that can fail should emit `destination.ErrorWithContext(...)` and stop emitting further values. For async work, wrap goroutines in `recoverUnhandledError` so panics surface through the configurable hooks.
- **Concurrency & backpressure**: Choose `NewSafeObservable`/`NewUnsafeObservable`/`NewEventuallySafeObservable` based on whether downstream code must be concurrency-safe. The safe variant serializes accesses; the unsafe variant favors throughput but requires callers to avoid concurrent emissions. `BackpressureDrop` is used to shed load when mutex acquisition would block.
- **Subscriptions**: Any operator that introduces additional resources (timers, channels, goroutines) must return a teardown that stops them. Use `sub := source.Subscribe...` and return `func(){ sub.Unsubscribe(); <cleanup> }` if multiple cleanups are needed.
- **Testing**: Tests rely on the fluent assertions in `testing/assert.go`. Pattern: `testing.Assert[T](t).Source(observable).ExpectNext(...).ExpectComplete().Verify()`. When writing tests for context-aware operators, prefer `VerifyWithContext` to control cancellation or deadlines.
- **Global hooks**: If you add code that might drop notifications or errors, be mindful of calling `OnDroppedNotification`/`OnUnhandledError` so users’ custom handlers are triggered.
- **Docs & Examples**: New APIs generally require documentation under `docs/` (Docusaurus) and an example under `examples/` following the workspace structure. Update `go.work` if you create a new module.

## External integrations
- External dependencies are primarily `github.com/samber/lo` for functional helpers and panic recovery, plus targeted libs inside plugins (HTTP client, fsnotify, etc.). Keep plugin dependencies isolated so the core module remains light.
- Some plugins demand newer Go releases (noted in `go.work`). Gate new code similarly if it depends on features from newer Go versions.

## Workflow reminders
- Run `go fmt`/`goimports` on touched Go files; tests and CI expect formatted code.
- When adding new modules or files, keep the Apache header and follow the established naming (`operator_<category>.go`, `<subject>_test.go`, etc.).
- Favor existing helper constructors (`NewObserverWithContext`, `NewSubscriberWithConcurrencyMode`, `recoverUnhandledError`) instead of rolling custom versions—the helpers bake in the library’s concurrency and error-handling invariants.
