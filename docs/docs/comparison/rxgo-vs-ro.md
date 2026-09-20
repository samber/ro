---
title: RxGo vs ro
description: Compare RxGo (ReactiveX/RxGo) and samber/ro — two ReactiveX-inspired implementations for Go, with and without generics
sidebar_position: 1
---

# ⚖️ RxGo vs `ro`

**RxGo and `ro` are both ReactiveX implementations for Go, but they were built a generation apart.** RxGo predates Go generics — its API is built on `interface{}` and runtime type assertions. `ro` requires Go 1.18+ and is generic from the ground up, so pipelines are type-checked at compile time instead of failing at runtime on a bad type assertion. If you already have RxGo in a codebase, there's no urgency to rewrite working code; for new Go 1.18+ projects, the generic-first design is `ro`'s main practical difference.

## Key differences

:::tip Core distinctions

### Type safety
- **RxGo**: Operators receive and return `interface{}`; you type-assert inside each callback (`item.(Customer)`). A mismatched assertion panics at runtime.
- **ro**: Operators are generic functions (`Map[T, R any]`). The compiler rejects a pipeline whose types don't line up — before you ever run it.

### API shape
- **RxGo**: Method-chaining on an `Observable` value (`observable.Filter(...).Map(...)`), consumed via a channel returned by `.Observe()`.
- **ro**: Standalone generic functions composed with `ro.Pipe`/`ro.PipeOp`, consumed via `.Subscribe(...)`.

### Ecosystem
- **RxGo**: One module, one set of operators.
- **ro**: A small dependency-free core plus ~35 separate plugin modules (JSON, CSV, HTTP, rate limiting, structured logging, `samber/hot` cache, and more), each its own `go.mod`.

:::

## Code comparison

**RxGo** (`interface{}`-based, [pattern verified against the RxGo README](https://github.com/ReactiveX/RxGo)):

```go
package main

import (
    "context"
    "fmt"

    "github.com/reactivex/rxgo/v2"
)

func main() {
    observable := rxgo.Just(1, 2, 3, 4, 5)().
        Filter(func(item interface{}) bool {
            return item.(int)%2 == 0
        }).
        Map(func(_ context.Context, item interface{}) (interface{}, error) {
            return fmt.Sprintf("even-%d", item.(int)), nil
        })

    for item := range observable.Observe() {
        fmt.Println(item.V) // "even-2", "even-4"
    }
}
```

**ro** (generics-based, compile-time checked):

```go
package main

import (
    "fmt"

    "github.com/samber/ro"
)

func main() {
    observable := ro.Pipe2(
        ro.Just(1, 2, 3, 4, 5),
        ro.Filter(func(x int) bool {
            return x%2 == 0
        }),
        ro.Map(func(x int) string {
            return fmt.Sprintf("even-%d", x)
        }),
    )

    observable.Subscribe(ro.OnNext(func(s string) {
        fmt.Println(s) // "even-2", "even-4"
    }))
}
```

The `ro` version never touches `interface{}`: `x` is `int` and the mapped value is `string`, enforced by the compiler at every step of the pipeline.

## When to use which

:::info Decision guide

### Stay with RxGo when:
- You have an existing RxGo codebase that works and isn't blocking you
- Your project targets a Go version older than 1.18 (no generics)
- Your team is already fluent in its `interface{}`-based API

### Use `ro` when:
- You're starting a new project on Go 1.18+ and want compile-time type safety
- You want a small core with opt-in plugins instead of one large dependency
- You need built-in context propagation through the whole pipeline (`SubscribeWithContext`, `NextWithContext`, ...)

:::

## Learn more

- [Getting started with ro](../getting-started)
- [samber/lo and samber/ro](./lo-vs-ro) — ro's sibling library for synchronous collections
- [channels vs ro](./channels-vs-ro)
- [Operators guide](../core/operators)
