---
name: Tail
slug: tail
sourceRef: operator_filter.go#L533
type: core
category: filtering
signatures:
  - "func Tail[T any]()"
playUrl:
variantHelpers:
  - core#filtering#tail
similarHelpers: [core#filtering#last, core#filtering#head, core#filtering#takelast]
position: 51
---

Emits only the last item emitted by an Observable. If the source Observable is empty, Tail will emit an error.

```go
obs := ro.Pipe[string, string](
    ro.Just("first", "second", "third"),
    ro.Tail(),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: third
// Completed
```

### With single item

```go
obs := ro.Pipe[string, string](
    ro.Just("only one"),
    ro.Tail(),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: only one
// Completed
```

### With empty observable

```go
obs := ro.Pipe[string, string](
    ro.Empty[string](),
    ro.Tail(),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Error: tail of empty observable
```

### With numbers

```go
obs := ro.Pipe[int, int](
    ro.Just(10, 20, 30, 40, 50),
    ro.Tail(),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 50
// Completed
```

### With time-based emissions

```go
obs := ro.Pipe[int64, int64](
    ro.Interval(100*time.Millisecond),
    ro.Tail(),
    ro.Take(3),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
time.Sleep(400 * time.Millisecond)
sub.Unsubscribe()

// Next: 2 (last value before completion)
// Completed
```

### With error in source

```go
obs := ro.Pipe[string, string](
    ro.Throw[string](errors.New("source error")),
    ro.Tail(),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Error: source error
```

### With large number of items

```go
obs := ro.Pipe[int, int](
    ro.Just(makeRange(1, 1000)...), // Emits 1 through 1000
    ro.Tail(),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1000
// Completed
```