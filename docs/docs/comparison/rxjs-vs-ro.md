---
title: RxJS vs ro
description: Coming from JavaScript/TypeScript? Compare RxJS and samber/ro — the same ReactiveX vocabulary, ported to Go's type system and concurrency model
sidebar_position: 2
---

# ⚖️ RxJS vs `ro`

**If you already know RxJS, you already know most of `ro`.** Both implement the same [ReactiveX](https://reactivex.io/) specification: Observables, Observers, Subscriptions, and a shared vocabulary of operators (`Map`, `Filter`, `Merge`, `CombineLatest`, `Debounce`...). The differences come from the host language, not the paradigm — Go's static typing and explicit context/cancellation model, versus TypeScript's structural typing and the browser/Node.js event loop.

## Key differences

:::tip Core distinctions

### Type system
- **RxJS**: TypeScript generics are optional and structural; `any` is always an escape hatch.
- **ro**: Go generics are mandatory and nominal — every operator's type parameters are checked at compile time, with no `any` escape hatch in the core API.

### Cancellation
- **RxJS**: `Subscription.unsubscribe()`, plus `AbortSignal` interop for newer APIs.
- **ro**: `Subscription.Unsubscribe()`, plus first-class `context.Context` propagation through every operator (`SubscribeWithContext`, `NextWithContext`, ...) — closer to how idiomatic Go already handles cancellation.

### Operator composition
- **RxJS**: `source$.pipe(filter(...), map(...))` — a fluent method chain.
- **ro**: `ro.Pipe2(source, ro.Filter(...), ro.Map(...))` — a plain function call, because Go has no fluent generic method chaining.

### Naming
- Most operator names carry over directly: `Map`, `Filter`, `Merge`, `Zip`, `CombineLatest`, `Take`, `Skip`, `Debounce`/`DebounceTime` → `ThrottleWhen`/`ThrottleTime` (see the [full operator reference](../operator/) for exact Go names and variants).

:::

## Code comparison

**RxJS**:

```ts
import { of } from 'rxjs';
import { filter, map } from 'rxjs/operators';

const observable$ = of(1, 2, 3, 4, 5).pipe(
  filter((x) => x % 2 === 0),
  map((x) => `even-${x}`),
);

observable$.subscribe((s) => console.log(s));
// "even-2", "even-4"
```

**ro**:

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

Same shape, same operators, same order of execution — the callback signatures are just statically typed instead of inferred.

## When to use which

:::info Decision guide

### Stay with RxJS when:
- You're building a frontend or Node.js service — `ro` is Go-only, it has no JS runtime
- Your team's reactive expertise is already in TypeScript

### Use `ro` when:
- You're porting reactive logic from a JS/TS service into a Go backend and want the same mental model
- You want compile-time guarantees RxJS's structural typing can't give you
- Your pipeline needs to respect Go's `context.Context` cancellation end-to-end

:::

## Learn more

- [Getting started with ro](../getting-started)
- [RxGo vs ro](./rxgo-vs-ro) — the historical ReactiveX implementation for Go
- [Operators reference](../operator/) — Go names for every RxJS operator you already know
- [Glossary](../glossary) — ReactiveX terminology, as used in `ro`
