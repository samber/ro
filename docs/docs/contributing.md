---
title: 🤝 Contributing
description: Join the community of contributors.
sidebar_position: 400
---

# 🤝 Contributing

Hey! We are happy to have you as a new contributor. ✌️

## Operator naming

Operators must be self-explanatory and respect standards (other languages, libraries...). Feel free to suggest many names in your contributions or the related issue.

`samber/ro` has been inspired by `ReactiveX` and `RxJS`. Find some inspiration in existing libraries:
- https://reactivex.io/documentation/operators.html
- https://reactivex.io/documentation/operators/buffer.html
- https://rxjs.dev/api

Many operators have variants. Please follow the same convention. Examples:

Map:
- Map: base operator
- MapI: the transformer function receives a forever increasing index
- MapWithContext: the transformer function receives a `context.Context`
- MapIWithContext: the transformer function receives a `context.Context` and a forever increasing index
- MapErr: the transformer function returns an error

Buffer:
- BufferWhen: the buffer is emitted on Observable notification
- BufferWithTime: the buffer is emitted when a timeout reached
- BufferWithCount: the buffer is emitted when size is reached
- BufferWithTimeOrCount: the buffer is emitted when a timeout or size is reached

Take:
- Take: emits N first items
- TakeWhile: emits items while a condition is met
- TakeUntil: emits items until a signal is sent over an Observable

Zip:
- Zip/ZipX/ZipAll/ZipWith/ZipWithX
- CombineLatest/CombineLatestX/CombineLatestAny/CombineLatestWith/CombineLatestWithX
- Merge/MergeAll/MergeWith/MergeWithX

...

We hate breaking changes, so better think twice ;)

## Context propagation in operators

`samber/ro` has been built with strict context propagation. New operators must not break the chain (propagation on subscription, message passing and unsubscription).

Example:
```go
func MapIWithContext[T, R any](project func(ctx context.Context, item T, index int64) (context.Context, R)) func(Observable[T]) Observable[R] {
    return func(source Observable[T]) Observable[R] {
        // This context has been provided by the downstream subscriber
        return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[R]) Teardown {
            i := int64(0)

            sub := source.SubscribeWithContext(
                // Subscribe to upstream with context received from downstream
                subscriberCtx,
                NewObserverWithContext(
                    func(ctx context.Context, value T) {
                        // The callback receives a context and return a new one (the same ?).
                        newCtx, result := project(ctx, value, i)
                        // Use .NextWithContext(...) instead of .Next(...)
                        destination.NextWithContext(newCtx, result)

                        i++
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

## Variadic operators

Many operators accept variadic parameters, providing flexibility while maintaining type safety:

Examples:
- `ro.Zip(...Observable[T])`
- `ro.ZipAll(...Observable[T])`
- `ro.Merge(...Observable[T])`
- `ro.MergeWith[T any](...Observable[T])`

## Type aliases on generics

Some operators use `~[]T` constraints to accept any slice type, including named slice types, not just `[]T`. This design choice makes the library more flexible in real-world usage.

Examples:
- `func Flatten[T any, Slice ~[]T]() func(Observable[Slice]) Observable[T]`

## Variants

When applicable, some operator might be declined in multiple ways. Update the documentation for each helper.

Examples:
- Map: base operator
- MapI: the transformer function receives a forever increasing index
- MapWithContext: the transformer function receives a `context.Context`
- MapIWithContext: the transformer function receives a `context.Context` and a forever increasing index
- MapErr: the transformer function returns an error
- MapErrI: the transformer function returns an error
- ...

## Testing

We try to maintain code coverage high.

Use the `ro.Collect(...)` for testing.

Example:
```go
values, err := Collect(
    Pipe1(
        Just([]int{1, 2, 3}, []int{4, 5, 6}),
        Flatten[int](),
    ),
)
is.Equal([]int{1, 2, 3, 4, 5, 6}, values)
is.NoError(err)
```

Test edge cases with `ro.Empty[int]()` and `ro.Throw[[]int](assert.AnError)` as source.

Example:
```go
values, err := Collect(
    Pipe1(
        Empty[[]int](),
        Flatten[int](),
    ),
)
is.Equal([]int{}, values)
is.NoError(err)

values, err = Collect(
    Pipe1(
        Throw[[]int](assert.AnError),
        Flatten[int](),
    ),
)
is.Equal([]int{}, values)
is.EqualError(err, assert.AnError.Error())
```

Test more edge cases:
- early unsubscription
- context propagation
- context cancellation

## Benchmark and performance

Write performant operators and limit extra memory consumption. Build an helper for general purpose and don't optimize for a particular use-case.

Feel free to write benchmarks.

Sources can be unbounded and might run for a very long time. If you expect a big memory footprint, please warn developers in the operator comment.

## Memory leaks

Streams can run forever, so any state that grows with the number of items, subscriptions or iterations is a leak.

- Return a teardown that releases every resource the subscription acquired: upstream subscription, timers, tickers, goroutines, channels.
- Stop every goroutine on unsubscription, completion and error. Tie it to `subscriberCtx` or a `done` channel.
- Declare state inside the subscribe callback, never outside: state shared between subscriptions outlives them.
- Bound buffers, queues, caches and maps, or warn in the operator comment when they cannot be bounded.
- Drop references (items, contexts, closures) once they are no longer needed. Clear slices and map entries instead of re-slicing.
- Call `Stop()` on timers and `cancel()` on derived contexts.
- Do not recurse without a bound: non-tail recursion on an infinite stream grows the stack.
- Test with `goleak` and a long run of subscribe/unsubscribe cycles.

## Higher-order Observables: races and memory leaks

Operators consuming an `Observable[Observable[T]]` (`MergeAll`, `ConcatAll`, `MergeMap`, `FlatMap`...) or resubscribing in a loop (`Retry`, `While`, `DoWhile`, `Repeat`...) create one inner subscription per outer item or iteration. On a long-lived or unbounded source, the number of inner subscriptions is unbounded too.

**Memory leaks.** Never keep every inner subscription in an aggregate that is only torn down when the operator ends. A finished inner subscription still pins its subscriber and closures, so memory grows with the number of inner Observables, not with the number of active ones.

- Release each inner subscription as soon as it completes, errors or is unsubscribed (remove it from the aggregate, or never add it when it is already closed).
- Keep only what is still active. Memory must be bounded by the number of concurrent inner Observables.
- Do not retain inner values, contexts or closures after the inner Observable is done.
- Buffered inner Observables (e.g. waiting in `ConcatAll`) are unbounded when the outer source is faster than the inner ones: document it in the operator comment.
- When the loop is synchronous and each subscription is already closed on return, there is nothing to aggregate: return a `nil` teardown.

**Races.** Outer and inner Observables may emit concurrently, and a new inner Observable may arrive while the downstream is unsubscribing.

- Guard shared state (active counter, aggregate, parent context) with a mutex or atomics.
- Count the outer Observable as an active source, so completion fires exactly once, when the outer and every inner Observable are done.
- Check `IsClosed()` before subscribing to a late inner Observable. Unsubscribe it immediately if the downstream is already closed.
- Never call `destination.Next*` from two goroutines at once without serialization (see `NewSafeObservable` and `Serialize`).

**Tests.** Run them with `-race` and `goleak`. Add a case with a large number of short-lived inner Observables, and one that unsubscribes while inner Observables are still emitting.

## Race condition patterns

Every new or changed operator MUST have a native `FuzzXxx(f *testing.F)` target for every applicable pattern below. Hand-rolled `for i := 0; i < N` hammer loops are not accepted for new tests.

**Sync and async sources.** Test BOTH kinds. A fuzz target that only uses synchronous sources is incomplete.

- A synchronous source emits inside `Subscribe`. The teardown is registered only after `Subscribe` returns, so it cannot stop the source.
- An asynchronous source emits from its own goroutine. `Next` races `Unsubscribe` and `Complete`.
- Let a bit of the fuzz input select the kind: use `fuzzSource` and `fuzzIsAsync`.

**Always run with `-race`.** Use `go test -race ./...`, `make test` or `make fuzz`. A race-free result without `-race` proves nothing.

**Inputs encode the interleaving, never the expected result.** Typical inputs: seed, goroutine count, item count, sync/async bitmask, unsubscribe-after index. The body asserts invariants:

- no overlapping `Next`
- at most one terminal notification, and nothing after it
- every upstream, signal and inner subscription released
- no goroutine leak
- no deadlock: every wait is bounded (`fuzzWaitFor`, `fuzzDeadline`)

**Seeds and iterations.**

- Register seeds with `xtest.AddSeeds`, so a plain `go test -race` explores them without `-fuzz`.
- One shared counter, `RO_FUZZ_ITERATIONS`, sets the seed count: 100 by default, 10 under `go test -short`.
- Run `make fuzz RO_FUZZ_ITERATIONS=10000` for a soak run.
- Commit crashers found by the fuzz engine under `testdata/fuzz/<FuzzName>/`.
- CI runs `make fuzz` on the stable Go version.
- Plugins are separate modules and cannot import `internal/xtest`. Copy the 15-line `fuzz_helpers_test.go` helper from `plugins/iter`, which reads the same env var.

| # | Pattern | Typical symptom | How to test |
|---|---------|-----------------|-------------|
| 1 | Teardown mutates shared state while a `Next` is in flight (`GroupBy` #436, `BufferWithCount` buffer reset) | Data race, lost or corrupted item | Async source, unsubscribe mid-stream, under `-race` |
| 2 | Shared state read outside the mutex, or a stale teardown resets a newer generation (#435, #423) | Data race, reconnect sees reset state | Concurrent subscribe/unsubscribe/reconnect cycles |
| 3 | Send on a closed channel at unsubscribe (#431, `ObserveOn`, `SubscribeOn`) | `panic: send on closed channel` | Async source, unsubscribe while items are in flight |
| 4 | Item popped under the lock but delivered after unlock (#420, `Zip`, `Buffer*`, `Window*`) | Concurrent `Complete` drops it, or items arrive out of order | Assert full, ordered output with async sources |
| 5 | Terminal lost or duplicated; late subscribe after a sync source already closed the destination (#432, #433, #430) | Hang, or two terminals | Count terminals; mix sync and async sources |
| 6 | Per-subscription state declared outside the subscribe callback (#427) | Subscriptions share counters or buffers | Subscribe the same Observable concurrently |
| 7 | Finished inner subscriptions retained (#424, #422) | Memory grows with inner count | Many short-lived inners; `activeCounter` back to 0 |
| 8 | Lock-free (unsafe) subscriber passed through an operator (`Catch`, `StartWith` over `Merge`) | Downstream gets overlapping `Next` | `serialGuard` in the observer, concurrent sources |
| 9 | Short-circuit operator keeps evaluating after its decision (#429) | Predicate called after the result is emitted | Count predicate calls after the terminal |
| 10 | Sync loop inside subscribe cannot be stopped (`Retry`, `While`, `DoWhile`, `RepeatWith`, `SubscribeOn`) | Unsubscribe never returns, infinite loop | Sync source, unsubscribe from the callback, bounded wait |
| 11 | Timer, ticker or goroutine outlives teardown, or ignores `subscriberCtx` | Goroutine leak, emission after teardown | `goleak`, cancel the context, assert nothing after |
| 12 | Lock held while emitting, so re-entrant calls deadlock (subjects, `Connect`, `UnicastSubject`) | Deadlock | Call `Next`/`Unsubscribe` from inside a callback, bounded wait |
| 13 | Unsubscription not propagated upstream (source, signal, inner, loser of a race) | Upstream keeps running | `trackSubscriptions` + `activeCounter`: count reaches 0 after `Unsubscribe`, `Complete` and `Error` |
| 14 | Deadlock: lock-order inversion, teardown waiting on a goroutine blocked in `Next`, `Wait`/`Collect` blocking after cancel (`Delay` `muQueue`/`muNext`) | Test hangs | Bounded waits on every blocking call |
| 15 | Long lock: `destination.Next` (or any callback, channel send, sleep) runs inside the operator mutex when only a state update needs it | Slow downstream blocks other sources and `Unsubscribe` | Slow observer + concurrent `Unsubscribe` with a short deadline |

For pattern 15, unlocking before emit can break ordering (pattern 4). Use a serializer (drain queue) rather than a plain unlock.

### Writing a fuzz target

```go
func FuzzMap(f *testing.F) {
	xtest.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := fuzzBound(seed, 0, fuzzMaxItems)
		unsubEarly := mask&0x80 != 0 // bit 7: unsubscribe right after Subscribe.
		up := &activeCounter{}
		guard := &serialGuard{}
		var terminals int32
		done := make(chan struct{})

		terminate := func() {
			if atomic.AddInt32(&terminals, 1) == 1 {
				close(done)
			}
		}

		source := trackSubscriptions(up, fuzzSource(seed, n, fuzzIsAsync(mask, 0)))
		sub := Map(func(v int) int { return v * 2 })(source).Subscribe(NewObserver(
			func(int) {
				guard.enter()
				defer guard.leave()

				if atomic.LoadInt32(&terminals) > 0 {
					t.Error("Next after terminal")
				}
			},
			func(error) { terminate() },
			terminate,
		))

		if unsubEarly {
			sub.Unsubscribe()
		} else {
			select {
			case <-done:
			case <-time.After(fuzzDeadline):
				t.Fatal("no terminal notification")
			}
		}

		fuzzWaitFor(t, "upstream released", func() bool { return up.activeCount() == 0 })

		if guard.overlapped() > 0 || atomic.LoadInt32(&terminals) > 1 {
			t.Fatalf("overlaps=%d terminals=%d", guard.overlapped(), atomic.LoadInt32(&terminals))
		}
	})
}
```

## Core vs plugins

**Never add a third-party library dependency to the core `ro` package.** If an operator requires wrapping an external library, it must live in a dedicated plugin under `plugins/` with its own `go.mod`. The core package only depends on `samber/lo`.

## Documentation

Operators must be properly commented, with a Go Playground link and a markdown documentation in `docs/data/`. In markdown header, please link to similar helpers (and update other markdowns accordingly).

Operator variants can be grouped in a single markdown.

New plugins must have their own page in `docs/docs/plugins/`.

Add your plugin or operator to `docs/static/llms.txt`.

## Examples

Create a [Go Playground](https://go.dev/play/) demonstration for each operator, allowing developers to quickly experiment and understand behavior without setting up a local environment.

Please add an example of your operator in the file named `ro_example_test.go`. It will be visible in Godoc website: https://pkg.go.dev/github.com/samber/ro

## Error conventions

Errors must be declared as package-level sentinel variables using `errors.New`, never as inline strings or `fmt.Errorf` calls in a `panic`.

Each package (core or plugin) that panics on invalid input **must** declare its errors in a dedicated `errors.go` file:

```go
// errors.go
package myplugin

import "errors"

var (
    ErrMyOperatorWrongParam = errors.New("myplugin.MyOperator: param must be greater than 0")
)
```

Then use the variable in the operator:

```go
func MyOperator(param int) func(ro.Observable[T]) ro.Observable[T] {
    if param <= 0 {
        panic(ErrMyOperatorWrongParam)
    }
    // ...
}
```

**Rules:**
- Use `errors.New` — never `fmt.Errorf` or a bare string — for sentinel error declarations.
- Error variable names follow the pattern `Err{OperatorName}{WhatIsWrong}` (e.g., `ErrRandomWrongSize`, `ErrWebsocketSubjectURLRequired`).
- Error messages follow the pattern `{package}.{FunctionName}: {lowercase description}` (e.g., `"rostrings.Random: size must be greater than 0"`).
- Never write `panic("some string")` or `panic(errors.New("..."))` inline — always use a pre-declared variable.
- Panics are reserved for programmer errors detected at construction time (invalid parameters), never for runtime stream errors.

## Upstream parity

Some operators re-implement algorithms from a sibling library (`samber/lo`); others simply wrap it. These two cases require different maintenance strategies.

### Mode 1 — Re-implemented code (manual sync)

`plugins/strings/operator_*.go` and `plugins/bytes/operator_*.go` contain operators whose logic is **copied from `github.com/samber/lo`** (`string.go`). The affected functions include `words`, `capitalize`, `pascalcase`, `camelcase`, `snakecase`, `kebabcase`, `ellipsis`, `random`, and the case-conversion helpers. Any bug fix or improvement in `samber/lo` must be ported manually to both plugins (code, tests, and doc).

Each file that copies upstream logic carries a header comment:

```go
// Ported from github.com/samber/lo (string.go) — keep in sync.
```

If you modify one of these operators and the change improves algorithm correctness (not just the reactive wrapping), check whether `samber/lo` has already applied the same fix, and vice-versa.

**Known divergence**: `bytes.ToLower` produces `U+FFFD` on invalid UTF-8, whereas `cases.Lower(...).Bytes()` preserves raw bytes. This is intentional; do not "fix" it to match `lo` without understanding the impact on byte-level consumers.

### Mode 2 — Wrapped library (go.mod bump)

These plugins **import** the sibling library and call its API; they do not copy logic. Synchronize by bumping the dependency version in `go.mod`:

- `plugins/samber/hot` → `github.com/samber/hot`
- `plugins/samber/psi` → `github.com/samber/psi`
- `plugins/iter` → iterator library
- `plugins/testify` → testify helpers
- `plugins/ozzo/ozzo-validation` → ozzo-validation

The **core `ro` module** also imports `samber/lo` (e.g., `lo.Must`), but only as a general utility — this is not a parity case.

### General rule

Before modifying an operator, determine whether it **re-implements** upstream logic (sync code + tests + doc, maintain the provenance comment) or **wraps** it (bump `go.mod`). Any file that copies upstream logic MUST carry `// Ported from … — keep in sync.`

## Other conventions

### Naming

1- If a callback returns a single bool then it should probably be called "predicate".
2- If a callback is used to change a collection element into something else then it should probably be called "transform".
3- If a callback returns nothing (void) then it should probably be called "callback".

### Types

1- Generic functions must preserve the underlying type of collections so that the returned values maintain the same type as the input. See [#365](https://github.com/samber/lo/pull/365/files).
