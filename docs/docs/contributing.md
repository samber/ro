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

Every new or changed operator MUST have a native `FuzzXxx(f *testing.F)` target for every applicable pattern below. Hand-rolled `for i := 0; i < N` hammer loops are not accepted.

**Test sync and async sources.** A fuzz target that only uses synchronous sources is incomplete.

- A synchronous source emits inside `Subscribe`. Your teardown is registered only after `Subscribe` returns, so it cannot stop that source.
- An asynchronous source emits from its own goroutine. Its `Next` races `Unsubscribe` and `Complete`.

Let a bit of the fuzz input pick the kind, with `fuzzSource` and `fuzzIsAsync`.

**Always run with `-race`.** Use `go test -race ./...`, `make test` or `make fuzz`. A clean run without `-race` proves nothing.

**Inputs encode the interleaving, never the expected result.** Typical inputs: seed, goroutine count, item count, sync/async bitmask, unsubscribe-after index. The body asserts invariants:

- no overlapping `Next`
- at most one terminal notification, and nothing after it
- every upstream, notifier and inner subscription released
- no goroutine leak
- no deadlock: every wait is bounded (`fuzzWaitFor`, `fuzzDeadline`)

**Where the targets and helpers live.** Core fuzz targets live in the `fuzz/` directory (`package fuzz`, same module, imports `github.com/samber/ro`). Its test-only file `fuzz/helpers_test.go` provides:

| Helper                                | Purpose                                                            |
| ------------------------------------- | ------------------------------------------------------------------ |
| `fuzzSource(seed, n, async)`          | Emits `0..n-1`, then completes, synchronously or from a goroutine  |
| `fuzzIsAsync(mask, i)`                | Reads bit `i` of a fuzz byte to pick sync or async                 |
| `fuzzBound(v, lo, hi)`                | Maps any fuzz integer into `[lo, hi]`                              |
| `fuzzJitter(seed, step)`              | Yields the scheduler at seed-chosen points                         |
| `serialGuard`                         | Counts overlapping callback calls (`enter`, `leave`, `overlapped`) |
| `activeCounter`, `trackSubscriptions` | Count live subscriptions to a source                               |
| `fuzzWaitFor`, `fuzzDeadline`         | Bounded polling and waiting                                        |

The same file runs `goleak` in `TestMain`, so a leaked goroutine fails the package.

**Seeds and iterations.**

- Register seeds with `xfuzz.AddSeeds` (`internal/xfuzz`), so a plain `go test -race` explores them without `-fuzz`.
- `RO_FUZZ_ITERATIONS` sets the seed count of every target: 100 by default, 10 under `go test -short`. `make fuzz` sets it to 1000.
- Run `make fuzz RO_FUZZ_ITERATIONS=10000` for a soak run.
- Commit crashers found by the fuzz engine under `testdata/fuzz/<FuzzName>/`.
- CI runs `make fuzz` on the stable Go version.
- Plugin fuzz tests import `github.com/samber/ro/internal/xfuzz` and call `xfuzz.AddSeeds`, with the same `RO_FUZZ_ITERATIONS` variable.

### How `ro` runs your operator

Most patterns below follow from four facts:

- **Teardown comes last.** The subscribe callback returns the teardown. With a synchronous source, every item and the terminal notification happen before the teardown exists.
- **`Unsubscribe` does not wait.** It runs the teardown while an asynchronous `Next` may still be executing your callback.
- **The constructor picks the locking.** `NewUnsafeObservableWithContext` hands you a lock-free `destination`. `NewObservableWithContext` and `NewSafeObservableWithContext` serialize calls to it, and hold that lock while the downstream callback runs.
- **`destination.IsClosed()` is the stop signal.** It turns true after a terminal notification or a downstream `Unsubscribe`, even before your subscribe callback returns. `destination` drops every notification after the first terminal one.

### Pattern index

Tests and reviews cite these patterns by number. Snippets qualify the library with `ro.`, as in a plugin; drop the prefix inside the core package. Most are fragments of the subscribe callback of the Skeleton in [hacking](./hacking).

| #   | Pattern                                                                  | Symptom                                                  |
| --- | ------------------------------------------------------------------------ | -------------------------------------------------------- |
| 1   | Teardown mutates state while a `Next` is in flight                       | Data race, lost or corrupted item                        |
| 2   | Shared state read outside its lock, or reset by a stale teardown         | Data race, a reconnection sees reset state               |
| 3   | Send on a channel closed by the teardown                                 | `panic: send on closed channel`                          |
| 4   | Item taken under the lock, emitted after unlock                          | Item dropped by a concurrent `Complete`, or out of order |
| 5   | Terminal notification lost or duplicated                                 | Hang, or two terminals                                   |
| 6   | Per-subscription state declared outside the subscribe callback           | Subscriptions share counters or buffers                  |
| 7   | Finished inner subscriptions retained                                    | Memory grows with the number of inner Observables        |
| 8   | Lock-free `destination` called from several goroutines                   | Overlapping `Next` downstream                            |
| 9   | Short-circuit operator keeps evaluating after its decision               | Callback runs after the result was emitted               |
| 10  | Synchronous loop inside the subscribe callback ignores the stop signal   | Infinite loop, `Subscribe` never returns                 |
| 11  | Timer, ticker or goroutine outlives the teardown, or ignores the context | Goroutine leak, emission after teardown                  |
| 12  | Lock held while emitting, re-entrant call deadlocks                      | Deadlock                                                 |
| 13  | Unsubscription not propagated upstream                                   | Upstream keeps running                                   |
| 14  | Lock-order inversion, or a teardown waiting on a blocked goroutine       | Test hangs                                               |
| 15  | Lock held during the whole emission, when only a state update needs it   | Slow downstream blocks other sources and `Unsubscribe`   |

### Pattern 1: teardown mutates state while `Next` is in flight

An operator that buffers items resets its buffer in the teardown. An asynchronous `Next` still appends to it, because `Unsubscribe` does not wait for in-flight callbacks.

```go
// BAD: Next and teardown touch buf without a lock.
buf := make([]T, 0, size)
sub := source.SubscribeWithContext(subscriberCtx, ro.NewObserverWithContext(
    func(ctx context.Context, value T) {
        buf = append(buf, value)
        if len(buf) == size {
            destination.NextWithContext(ctx, buf)
            buf = make([]T, 0, size)
        }
    },
    destination.ErrorWithContext,
    destination.CompleteWithContext,
))

return func() {
    sub.Unsubscribe() // returns while an async Next may still run
    buf = nil         // BAD: data race with append
}
```

```go
// GOOD: one lock guards buf in Next and in the teardown.
var mu sync.Mutex
buf := make([]T, 0, size)
sub := source.SubscribeWithContext(subscriberCtx, ro.NewObserverWithContext(
    func(ctx context.Context, value T) {
        mu.Lock()
        if buf == nil { // teardown already ran: drop the item
            mu.Unlock()
            return
        }
        buf = append(buf, value)
        var full []T
        if len(buf) == size {
            full, buf = buf, make([]T, 0, size)
        }
        mu.Unlock()

        if full != nil {
            destination.NextWithContext(ctx, full) // emit outside the lock (pattern 12)
        }
    },
    destination.ErrorWithContext,
    destination.CompleteWithContext,
))

return func() {
    sub.Unsubscribe()
    mu.Lock()
    buf = nil
    mu.Unlock()
}
```

**How to test:** async source, unsubscribe after a fuzzed number of items. `-race` reports the race.

### Pattern 2: shared state outside its lock, or reset by a stale teardown

An operator that shares one upstream connection between subscribers keeps it in a struct. Two mistakes are common: reading a field without the lock, and a teardown of connection N resetting connection N+1.

```go
// BAD
func (s *shared[T]) connect(source ro.Observable[T], destination ro.Observer[T]) ro.Teardown {
    if s.conn == nil { // BAD: read outside the lock
        s.mu.Lock()
        s.conn = source.Subscribe(destination)
        s.mu.Unlock()
    }

    return func() {
        s.mu.Lock()
        s.conn = nil // BAD: may reset a newer connection
        s.mu.Unlock()
    }
}
```

```go
// GOOD: every access holds the lock, and a teardown only resets its own generation.
func (s *shared[T]) connect(source ro.Observable[T], destination ro.Observer[T]) ro.Teardown {
    s.mu.Lock()
    s.gen++
    gen := s.gen
    s.mu.Unlock()

    conn := source.Subscribe(destination)

    s.mu.Lock()
    if s.gen == gen {
        s.conn = conn
    }
    s.mu.Unlock()

    return func() {
        s.mu.Lock()
        if s.gen == gen {
            s.conn = nil
        }
        s.mu.Unlock()

        conn.Unsubscribe()
    }
}
```

**How to test:** concurrent subscribe, unsubscribe and reconnect cycles. Assert the state after the last cycle, under `-race`.

### Pattern 3: send on a channel closed by the teardown

An operator that hands items to a worker goroutine closes the channel in the teardown. An in-flight `Next` then sends on a closed channel and panics.

**Rule:** never close a channel that a callback sends on. Close a separate `done` channel and `select` on both.

```go
// BAD
items := make(chan T)
// ... a worker goroutine ranges over items and calls destination.NextWithContext
sub := source.SubscribeWithContext(subscriberCtx, ro.NewObserverWithContext(
    func(ctx context.Context, value T) { items <- value }, // panics once items is closed
    destination.ErrorWithContext,
    destination.CompleteWithContext,
))

return func() {
    close(items) // BAD: an in-flight Next may still send
    sub.Unsubscribe()
}
```

```go
// GOOD: jobs is never closed; done tells both sides to stop.
// Terminal notifications use the same channel, so they cannot overtake items (pattern 4).
jobs := make(chan func())
done := make(chan struct{})

go func() {
    for {
        select {
        case <-done:
            return
        case job := <-jobs:
            job()
        }
    }
}()

send := func(job func()) {
    select {
    case jobs <- job:
    case <-done: // teardown ran: drop the notification
    }
}

sub := source.SubscribeWithContext(subscriberCtx, ro.NewObserverWithContext(
    func(ctx context.Context, value T) { send(func() { destination.NextWithContext(ctx, value) }) },
    func(ctx context.Context, err error) { send(func() { destination.ErrorWithContext(ctx, err) }) },
    func(ctx context.Context) { send(func() { destination.CompleteWithContext(ctx) }) },
))

return func() {
    close(done)
    sub.Unsubscribe()
}
```

**How to test:** async source, unsubscribe while items are in flight. The panic fails the target.

### Pattern 4: item taken under the lock, emitted after unlock

An operator that queues items from several sources pops one under the lock, unlocks, then emits it. Between unlock and emit, another goroutine can emit the next item first, or send `Complete`, which drops this item.

```go
// BAD
mu.Lock()
item := queue[0]
queue = queue[1:]
mu.Unlock()

// another goroutine can pop and emit the next item before this line,
// or call Complete, which drops this item
destination.NextWithContext(ctx, item)
```

**Rule:** route items and the terminal notification through one ordered path. Either emit under the lock (simple, but see patterns 12 and 15), or use a drain queue (see the `serializer` in pattern 15).

**How to test:** async sources, assert the full output in the expected order, and that no item is missing before `Complete`.

### Pattern 5: terminal notification lost or duplicated

An operator that subscribes to several sources completes when the last one completes. A plain counter races between sources, so `Complete` fires twice or never. A synchronous source can also close `destination` before the next source is subscribed.

```go
// BAD
active := len(sources)
for _, source := range sources {
    // BAD: a synchronous source may already have closed destination
    subs.AddUnsubscribable(source.SubscribeWithContext(subscriberCtx, ro.NewObserverWithContext(
        destination.NextWithContext,
        destination.ErrorWithContext,
        func(ctx context.Context) {
            active-- // BAD: data race between sources
            if active == 0 {
                destination.CompleteWithContext(ctx)
            }
        },
    )))
}
```

```go
// GOOD
active := int32(len(sources))
if active == 0 {
    destination.CompleteWithContext(subscriberCtx) // no source still means one terminal
    return nil
}

subs := ro.NewSubscription(nil)
for _, source := range sources {
    if destination.IsClosed() { // a source already terminated destination
        break
    }

    subs.AddUnsubscribable(source.SubscribeWithContext(subscriberCtx, ro.NewObserverWithContext(
        destination.NextWithContext,
        destination.ErrorWithContext,
        func(ctx context.Context) {
            if atomic.AddInt32(&active, -1) == 0 { // exactly one source sees 0
                destination.CompleteWithContext(ctx)
            }
        },
    )))
}

return subs.Unsubscribe
```

**How to test:** count terminal notifications. Mix sync and async sources, and sources that error.

### Pattern 6: per-subscription state outside the subscribe callback

The operator function and the function returned by it run once. The subscribe callback runs once per subscription. State declared outside it is shared by every subscription.

```go
// BAD
return func(source ro.Observable[T]) ro.Observable[T] {
    count := 0 // shared by every subscription

    return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
        // ... count++ in Next
    })
}
```

```go
// GOOD
return func(source ro.Observable[T]) ro.Observable[T] {
    return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
        count := 0 // one counter per subscription
        // ... count++ in Next
    })
}
```

**How to test:** subscribe to the same Observable from several goroutines. `-race` reports the shared state, and per-subscription output differs.

### Pattern 7: finished inner subscriptions retained

An operator that subscribes to one inner Observable per outer item adds every inner subscription to one aggregate. Finished inners stay referenced until the operator ends, so memory grows with the number of inner Observables, not with the number of active ones.

```go
// BAD: keeps every inner subscription until the operator ends.
all.AddUnsubscribable(inner.SubscribeWithContext(ctx, destination))
```

```go
// GOOD: keep only active inners, and forget each one as soon as it ends.
mu.Lock()
id := nextID
nextID++
mu.Unlock()

release := func() {
    mu.Lock()
    delete(active, id)
    mu.Unlock()
}

sub := inner.SubscribeWithContext(ctx, ro.NewObserverWithContext(
    destination.NextWithContext,
    func(ctx context.Context, err error) { release(); destination.ErrorWithContext(ctx, err) },
    func(ctx context.Context) { release() }, // completion counting omitted (pattern 5)
))

mu.Lock()
if !sub.IsClosed() { // an inner that already ended is never stored
    active[id] = sub
}
mu.Unlock()
```

The teardown copies `active` under the lock, then unsubscribes each entry outside it. See also the section "Higher-order Observables: races and memory leaks" above.

**How to test:** many short-lived inner Observables wrapped with `trackSubscriptions`. Assert `activeCounter.activeCount()` returns to 0.

### Pattern 8: lock-free `destination` called from several goroutines

`NewUnsafeObservableWithContext` gives a `destination` without a lock. An operator that forwards several sources to it lets two goroutines call `Next` at the same time. The same happens when an operator passes its lock-free `destination` to a source that emits concurrently.

```go
// BAD: destination is lock-free, but several sources call it concurrently.
return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
    subs := ro.NewSubscription(nil)
    for _, source := range sources {
        subs.AddUnsubscribable(source.SubscribeWithContext(subscriberCtx, destination))
    }

    return subs.Unsubscribe
})
```

```go
// GOOD: the safe constructor serializes calls to destination.
return ro.NewSafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
    // same body
})
```

**Rule:** use the unsafe constructor only when a single goroutine calls `destination` at a time.

**How to test:** wrap the observer body with `serialGuard`, use concurrent async sources, assert `overlapped() == 0`.

### Pattern 9: short-circuit operator keeps evaluating after its decision

An operator that emits a result as soon as one item matches keeps calling the user callback on later items. A synchronous source keeps pushing them, because the teardown that would stop it does not exist yet.

```go
// BAD: predicate still runs after the decision.
func(ctx context.Context, value T) {
    if predicate(value) {
        destination.NextWithContext(ctx, true)
        destination.CompleteWithContext(ctx)
    }
},
```

```go
// GOOD: a flag stops evaluation once the result is emitted.
func(ctx context.Context, value T) {
    if atomic.LoadInt32(&decided) == 1 {
        return
    }
    if predicate(value) && atomic.CompareAndSwapInt32(&decided, 0, 1) {
        destination.NextWithContext(ctx, true)
        destination.CompleteWithContext(ctx)
    }
},
```

The `Complete` handler emits the fallback result only when `CompareAndSwapInt32(&decided, 0, 1)` succeeds.

**How to test:** count predicate calls after the terminal notification. Expect 0, with sync and async sources.

### Pattern 10: synchronous loop ignores the stop signal

An operator that loops inside the subscribe callback (emits generated values, or resubscribes to its source after each completion or error) runs before its teardown exists. Only `destination.IsClosed()` can stop it.

```go
// BAD: the teardown is registered only after this loop returns, which never happens.
return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
    for {
        destination.NextWithContext(subscriberCtx, generate())
    }
})
```

```go
// GOOD: a downstream Unsubscribe or a terminal notification closes destination,
// even before this callback returns.
return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
    for !destination.IsClosed() {
        destination.NextWithContext(subscriberCtx, generate())
    }

    return nil
})
```

**Rule:** check `destination.IsClosed()` before every iteration and before every new subscription.

**How to test:** sync source. Subscribe with a `Subscriber` and call its `Unsubscribe` from inside the `Next` callback. Bound the wait with `fuzzDeadline`.

### Pattern 11: timer, ticker or goroutine outlives the teardown

An operator that emits on a timer starts a goroutine or a ticker. Without a stop path, it leaks and may emit after the teardown.

```go
// BAD: never stopped, ignores cancellation.
go func() {
    for range time.Tick(interval) {
        destination.NextWithContext(subscriberCtx, value)
    }
}()
```

```go
// GOOD: stopped by the teardown and by context cancellation.
ticker := time.NewTicker(interval)
done := make(chan struct{})

go func() {
    for {
        select {
        case <-done:
            return
        case <-subscriberCtx.Done():
            return
        case <-ticker.C:
            destination.NextWithContext(subscriberCtx, latest()) // latest reads state under a lock
        }
    }
}()

// ... subscribe to source

return func() {
    close(done)
    ticker.Stop()
    sub.Unsubscribe()
}
```

**How to test:** `goleak` (already in `TestMain`). Cancel the context or unsubscribe, then assert nothing is emitted afterwards.

### Pattern 12: lock held while emitting

Downstream code may call back into the operator from inside `Next`: unsubscribe, or push a new item. If the operator still holds its `sync.Mutex`, the re-entrant call locks it again and the goroutine deadlocks on itself.

```go
// BAD
mu.Lock()
defer mu.Unlock()

state = append(state, value)
destination.NextWithContext(ctx, value) // downstream unsubscribes -> teardown locks mu -> deadlock
```

```go
// GOOD: update state under the lock, emit after unlock.
mu.Lock()
state = append(state, value)
mu.Unlock()

destination.NextWithContext(ctx, value)
```

Emitting after unlock can reorder items from several goroutines (pattern 4). In that case, use the `serializer` of pattern 15.

**How to test:** call `Unsubscribe` or `Next` from inside the observer callback. Bound the wait with `fuzzDeadline`.

### Pattern 13: unsubscription not propagated upstream

An operator that subscribes to a source and a `notifier` must release both on `Unsubscribe`, `Complete` and `Error`. A dropped subscription keeps the upstream running.

```go
// BAD: the notifier subscription is dropped, so nothing ever releases it.
notifier.SubscribeWithContext(subscriberCtx, notifierObserver)

return source.SubscribeWithContext(subscriberCtx, destination).Unsubscribe
```

```go
// GOOD: release every upstream.
notifierSub := notifier.SubscribeWithContext(subscriberCtx, notifierObserver)
if destination.IsClosed() { // a synchronous notifier already ended the stream
    return notifierSub.Unsubscribe
}

sourceSub := source.SubscribeWithContext(subscriberCtx, destination)

return func() {
    notifierSub.Unsubscribe()
    sourceSub.Unsubscribe()
}
```

`destination` runs the teardown after a terminal notification too, so one teardown covers the three cases.

**How to test:** wrap each upstream with `trackSubscriptions`. Assert `activeCount()` reaches 0 after `Unsubscribe`, `Complete` and `Error`.

### Pattern 14: deadlock

Three shapes recur:

- **Lock-order inversion.** `Next` takes `muA` then `muB`, the teardown takes `muB` then `muA`. Use one lock, or always the same order.
- **Teardown waits on a blocked goroutine.** The teardown waits for a worker that is blocked in `destination.Next`, or that called the teardown itself.
- **Blocking wait after cancellation.** A wait on a subscription or a channel that nothing closes once the context is canceled.

```go
// BAD: hangs when the worker triggered this teardown from inside Next.
return func() {
    close(done)
    wg.Wait()
}
```

**Rule:** a teardown signals goroutines to stop. It never waits for them.

**How to test:** bound every blocking call with `fuzzWaitFor` or `fuzzDeadline`. Unsubscribe from inside callbacks and from other goroutines.

### Pattern 15: lock held during the whole emission

An operator holds its lock around `destination.Next` when only the state update needs it. A slow downstream then blocks every other source and `Unsubscribe`.

```go
// BAD
mu.Lock()
count++
destination.NextWithContext(ctx, value) // a slow downstream blocks every source and the teardown
mu.Unlock()
```

A plain unlock before emitting breaks ordering (pattern 4). Use a drain queue: update state and enqueue under the lock, then let one goroutine deliver the queue outside the lock.

```go
// GOOD: a drain queue keeps order and never emits under the lock.
type serializer struct {
    mu       sync.Mutex
    queue    []func()
    draining bool
}

// emit runs update under the lock and queues the job it returns, so the job order
// matches the update order. The first caller drains the queue outside the lock.
func (s *serializer) emit(update func() (job func())) {
    s.mu.Lock()
    if job := update(); job != nil {
        s.queue = append(s.queue, job)
    }
    if s.draining { // another goroutine is draining: it will run the job
        s.mu.Unlock()
        return
    }

    s.draining = true
    for len(s.queue) > 0 {
        job := s.queue[0]
        s.queue[0] = nil
        s.queue = s.queue[1:]

        s.mu.Unlock()
        job()
        s.mu.Lock()
    }
    s.draining = false
    s.mu.Unlock()
}
```

Declare one `serializer` per subscription (pattern 6). Send `Next`, `Error` and `Complete` through it:

```go
s.emit(func() func() {
    count++ // state update, under the lock
    return func() { destination.NextWithContext(ctx, value) }
})
```

A re-entrant `emit` from inside a job only enqueues, so it cannot deadlock. The queue is unbounded when sources outpace the downstream: document it in the operator comment.

**How to test:** slow observer, concurrent async sources and a concurrent `Unsubscribe` with a short deadline. Assert order with `serialGuard` at 0.

### Writing a fuzz target

Name the file after the source file it tests:

- Core: `fuzz/<root source file without .go>_fuzz_test.go`, e.g. `fuzz/operator_filter_fuzz_test.go` for `operator_filter.go`.
- Plugins: `<source file without .go>_fuzz_test.go` next to it, in the plugin directory, e.g. `plugins/<name>/source_fuzz_test.go` for `source.go`.

The target below lives in `fuzz/` (`package fuzz`), so it qualifies the library with `ro.`. It covers sync and async sources, early unsubscription, overlapping `Next`, duplicated terminals and upstream release.

```go
func FuzzMyOperator(f *testing.F) {
    // Seeds run under a plain `go test -race`, without -fuzz.
    xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i), uint8(i * 7)} })

    f.Fuzz(func(t *testing.T, seed int64, mask uint8, stopAt uint8) {
        n := fuzzBound(seed, 0, fuzzMaxItems)
        async := fuzzIsAsync(mask, 0)                     // bit 0: sync or async source
        stop := fuzzBound(int64(stopAt), 0, fuzzMaxItems) // unsubscribe after `stop` items when stop < n

        upstream := &activeCounter{}
        guard := &serialGuard{}
        var received, terminals int32
        done := make(chan struct{})
        terminate := func() {
            if atomic.AddInt32(&terminals, 1) == 1 {
                close(done)
            }
        }

        source := trackSubscriptions(upstream, fuzzSource(seed, n, async))
        sub := ro.MyOperator[int]()(source).Subscribe(ro.NewObserver(
            func(int) {
                guard.enter() // detects overlapping Next
                defer guard.leave()
                if atomic.LoadInt32(&terminals) > 0 {
                    t.Error("Next after terminal")
                }
                atomic.AddInt32(&received, 1)
            },
            func(error) { terminate() },
            terminate,
        ))

        if stop < n { // unsubscribe while items may still be in flight
            fuzzWaitFor(t, "items before stop", func() bool {
                return int(atomic.LoadInt32(&received)) >= stop || atomic.LoadInt32(&terminals) > 0
            })
            sub.Unsubscribe()
        } else {
            select {
            case <-done:
            case <-time.After(fuzzDeadline):
                t.Fatal("no terminal notification")
            }
        }

        fuzzWaitFor(t, "upstream released", func() bool { return upstream.activeCount() == 0 })
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
