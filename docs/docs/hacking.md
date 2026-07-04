---
title: 🏴‍☠️ Hacking
description: Extend ro with your own operators and plugins
sidebar_position: 200
---

# 🏴‍☠️ Extend ro with your own operators and plugins

This documentation is dedicated to developers implementing their own custom operators and plugins. If you just want to contribute to `samber/ro`, visit the [contributing section](./contributing).

:::tip Operators

For advanced demonstrations, please check the `samber/ro` source code.

:::

Remember that a stream might run indefinitely. You should pay attention to memory leaks, non-tail recursive function calls, retry mechanism, dangerous side-effects...

## Basic Operator abstraction

The simplest way to create custom operators is by composing existing ones:

```go
// Custom operator that doubles and filters even numbers
func DoubleAndFilterEven[T constraints.Integer]() func(ro.Observable[T]) ro.Observable[T] {
    return ro.PipeOp2(
        ro.Map(func(x T) T { return x * 2 }),
        ro.Filter(func(x T) bool { return x%2 == 0 }),
    )
}
```

## Skeleton

Here is a commented skeleton of your next operator:

```go
func MyOperator[T, R any](param1 int, param2 string) func(Observable[T]) Observable[R] {
    // Your code here will be executed once.
    // Useful for validating operator parameters.
    return func(source Observable[T]) Observable[R] {
        // Your code here will be executed once, when the operator is applied to an observable.
        // Don't code here.
        return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[R]) Teardown {
            // Your code here will be executed lazily, on every subscription.

            sub := source.SubscribeWithContext(subscriberCtx, ro.NewObserverWithContext(
                func(ctx context.Context, value T) {
                    // your code here will be executed for each message

                    // destination.NextWithContext(ctx, ...)
                },
                func(ctx context.Context, err error) {
                    // your code here will be executed at most once, when an error occurs

                    // destination.ErrorWithContext(ctx, error)
                },
                func(ctx context.Context) {
                    // your code here will be executed at most once, on stream completion

                    // destination.CompleteWithContext(ctx)
                },
            ))

            return func() {
                // Your code here will be executed on completion, error or early unsubscription.
                // Useful for cleaning resource of an active pipeline.
                // This might be run concurrently with the code above.

                sub.Unsubscribe()
            }
        })
    }
}
```

## Stateful operator

Each subscription creates a new state. The state must be declard in the Observable. If created outside, the state will be shared between subscriptions.

```go
func Scan[T, R any](initial R, accumulator func(R, T) R) func(Observable[T]) Observable[R] {
    return func(source Observable[T]) Observable[R] {
        return NewUnsafeObservable(func(destination Observer[R]) Teardown {
            // State 👇
            state := initial

            sub := source.Subscribe(ro.NewObserver(
                func(value T) {
                    state = accumulator(state, value)
                    destination.Next(state)
                },
                destination.Error,
                destination.Complete,
            ))

            return sub.Unsubscribe
        })
    }
}
```

## Safe vs unsafe Observable

Unsafe observables are much faster but offer less protection against race conditions. Use `ro.NewSafeObservable` if you expect asynchronous behavior in the callback, or `ro.NewUnsafeObservable` of inner code is synchronous.

```go
// Note: This is a creation operator, not chainable operator
func AsyncHTTPRequest(req *http.Request) ro.Observable[*http.Response] {
    // A "safe" observable prevents concurrent message passing through destination.Next()
    return ro.NewSafeObservable(func(destination ro.Observer[*http.Response]) ro.Teardown {
        ctx, cancel := context.WithCancel(req.Context())

        go func() {
            req = req.WithContext(ctx)

            res, err := http.DefaultClient.Do(req)
            if err != nil {
                destination.ErrorWithContext(ctx, err)
                return
            }

            destination.NextWithContext(ctx, res)
            destination.CompleteWithContext(ctx)
        }()

        // the request will be canceled on early unsubscription
        return func () {
            cancel()
        }
    })
}
```

```go
// Note: This is a creation operator, not chainable operator
func SyncHTTPRequest(req *http.Request) ro.Observable[*http.Response] {
    // An "unsafe" observable is not protected against concurrent message passing through destination.Next()
    return ro.NewUnsafeObservable(func(destination ro.Observer[*http.Response]) ro.Teardown {
        req = req.WithContext(context.Background())

        res, err := http.DefaultClient.Do(req)
        if err != nil {
            destination.ErrorWithContext(ctx, err)
            return nil
        }

        destination.NextWithContext(ctx, res)
        destination.CompleteWithContext(ctx)

        // The request is already ended. No need to return canceler.
        return nil
    })
}
```

## Error handling

Always handle errors properly in custom operators. Unhandled errors can cause memory leaks or undefined behavior.

```go
func SafeMap[T, R any](mapper func(T) (R, error)) func(ro.Observable[T]) ro.Observable[R] {
    return func(source ro.Observable[T]) ro.Observable[R] {
        return ro.NewObservable(func(observer ro.Observer[R]) ro.Teardown {
            sub := source.Subscribe(ro.NewObserver(
                func(value T) {
                    result, err := mapper(value)
                    if err != nil {
                        // Graceful stop of the stream
                        observer.Error(err)
                        return
                    }
                    observer.Next(result)
                },
                observer.Error,
                observer.Complete,
            ))
            return sub.Unsubscribe
        })
    }
}
```

## Resource cleanup

Always clean up resources properly. Use teardown functions to prevent memory leaks.

```go
func WithTimeout[T any](timeout time.Duration) func(ro.Observable[T]) ro.Observable[T] {
    return func(source ro.Observable[T]) ro.Observable[T] {
        return ro.NewObservable(func(observer ro.Observer[T]) ro.Teardown {
            timer := time.NewTimer(timeout)
            done := make(chan struct{})

            subscription := source.Subscribe(ro.NewObserver(
                func(value T) {
                    select {
                    case <-timer.C:
                        observer.Error(fmt.Errorf("timeout"))
                    default:
                        observer.Next(value)
                    }
                },
                observer.Error,
                observer.Complete,
            ))

            return func() {
                // Called on error, completion or unsubscription
                close(done)
                timer.Stop()
                subscription.Unsubscribe()
            }
        })
    }
}
```

## Context propagation in operators

`samber/ro` has been built with strict context propagation. Your operators must not break the chain (propagation on subscription, message passing and unsubscription).

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

## Other Considerations

- Your operator may check its Subscriber’s `IsClosed()` status before it emits any item to (or sends any notification to) the Subscriber. Do not waste time generating items that no Subscriber is interested in seeing.
- Your operator should obey the core tenets of the Observable contract:
  - It may call a Subscriber’s `Next(...)` method any number of times, but these calls must be non-overlapping.
  - It may call either a Subscriber’s `Completed()` or `Error()` method, but not both, exactly once, and it may not subsequently call a Subscriber’s onNext(...) method.
  - If you are unable to guarantee that your operator conforms to the above two tenets, you can use `NewSafeObservable` or the `Serialize()` operator to it to force the correct behavior.
- Do not block within your operator.
- It is usually best that you compose new operators by combining existing ones, to the extent that this is possible, rather than reinventing the wheel. `ro` itself does this with some of its standard operators, for example:
- If your operator uses functions that are passed in as parameters (predicates, for instance), note that these may be sources of errors, and be prepared to catch these and notify subscribers via `Error()` calls.
- In general, notify subscribers of error conditions immediately, rather than making an effort to emit more items first.

## Add / port an operator end-to-end

Follow this checklist in order when adding a new operator or porting one from a sibling library. Each step maps to a concrete file in the repository.

1. **Operator code + variants**: implement the base operator and all applicable variants in the correct suffix order: `Err` → `I` → `WithContext` (e.g., `Map`, `MapErr`, `MapI`, `MapErrI`, `MapWithContext`, `MapErrWithContext`, `MapIWithContext`, `MapErrIWithContext`).
2. **Error sentinels**: if the operator validates its parameters, declare sentinel variables in `errors.go` using the pattern `Err{OperatorName}{WhatIsWrong}`. Never `panic("string")` inline.
3. **Tests**: write tests using `Collect(...)` with at minimum:
   - `Empty[T]()` (empty source)
   - `Throw[T](assert.AnError)` (error propagation)
   - Early unsubscription
   - Context propagation and cancellation
4. **Godoc example**: add an example function in `ro_example_test.go` (core) or `plugins/<x>/operator_example_test.go` (plugin). It will appear on https://pkg.go.dev.
5. **Go Playground link**: create a runnable snippet via the `mcp__go-playground__run_and_share_go_code` MCP tool, verify it executes correctly, then add `// Play: https://go.dev/play/p/...` above the function signature. Leave the URL empty if the operator is not yet published (new code not yet released cannot compile on the Playground).
6. **Markdown doc**: create `docs/data/(core|plugin)-<name>.md` with complete frontmatter (`sourceRef`, `signatures`, `variantHelpers`, `similarHelpers`, `playUrl`, `position`). See [`docs/CLAUDE.md`](../CLAUDE.md) for the full format.
7. **llms.txt**: add a one-line entry for the operator in `docs/static/llms.txt`.
8. **Upstream parity**: if the operator re-implements logic from a sibling library, synchronize it. See [Upstream parity](./contributing#upstream-parity) in contributing.md.
9. **Verify**: run `make test` (race detector) and `make lint`. Documentation validation scripts run in CI — you do not need to run them manually.

## Common agent pitfalls

These issues have recurred across multiple sessions. Read them before starting.

- **`PrintObserver[[]byte]()`** prints raw integer slices (`[196 176 ...]`), not strings. For byte-slice observables, write a custom observer that calls `fmt.Println(string(data))`.
- **`context.TODO()` in goroutines breaks the context chain.** Any goroutine spawned inside a plugin operator must capture and forward `subscriberCtx`, not create an independent context. Using `context.TODO()` silently breaks cancellation and deadline propagation.
- **Adding an import before its first usage** causes `imported and not used` compile errors. Add the import and its usage in the same edit.
- **`go.work.sum` churn**: the workspace sum file is modified by routine `go` tool invocations. Do not commit it unless it is the only intentional change. Restore with `git checkout go.work.sum` when it appears as an unintentional modification.
- **Porting from `samber/lo` — bytes UTF-8 divergence**: `bytes.ToLower` via `strings.ToLower` produces `U+FFFD` replacement characters on invalid UTF-8, whereas `cases.Lower(...).Bytes()` preserves the raw bytes. Adjust test fixtures accordingly when porting from `samber/lo` string helpers to the `bytes` plugin.

## Definition of done

Before opening a PR, confirm:

```bash
make test   # runs all tests with the race detector
make lint   # runs golangci-lint + license header check (make lint-fix to auto-correct)
```

To test a single plugin module without running the whole workspace:

```bash
cd plugins/<x> && go test -race ./...
```

**Known pre-existing failures — do not confuse with your own changes:**

- `bench/` and `plugins/exp/simd/` fail under a plain `go build`/`go test` because `exp/simd` requires `GOEXPERIMENT=simd GOWORK=off`.
- Several plugins are commented out of `go.work` because they require a newer Go version than the workspace default: `cron`, `exp/simd`, `encoding/json/v2`, `ics`, `hyperloglog`, `iter`, `sentry`, `slog`, `zap`, `oops`, `hot`. They are excluded from `make test`.

## Next Steps

- **[Operators Reference](./operator/)** - Learn about existing operators for inspiration
- **[Testing Guide](./testing)** - Learn how to test your custom operators
- **[Subject Documentation](./core/subject)** - Understand subjects for custom implementations
- **[Troubleshooting Guide](./troubleshooting/)** - Debug issues with custom operators

Happy hacking! 🏴‍☠️
