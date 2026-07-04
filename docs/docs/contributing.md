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
