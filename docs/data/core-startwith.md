---
name: StartWith
slug: startwith
sourceRef: operator_combining.go#L906
type: core
category: combining
signatures:
  - "func StartWith[T any](prefixes ...T)"
playUrl:
variantHelpers:
  - core#combining#startwith
similarHelpers:
  - core#combining#endwith
  - core#combining#concat
  - core#creation#just
position: 75
---

Emits the given values before emitting the values from the source Observable.

```go
obs := ro.Pipe[string, string](
    ro.Just("a", "b", "c"),
    ro.StartWith("x", "y"),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: x
// Next: y
// Next: a
// Next: b
// Next: c
// Completed
```

### With single prefix value

```go
obs := ro.Pipe[int, int](
    ro.Just(2, 3, 4),
    ro.StartWith(1),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Next: 4
// Completed
```

### With multiple prefix values

```go
obs := ro.Pipe[string, string](
    ro.Just("c"),
    ro.StartWith("a", "b"),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: a
// Next: b
// Next: c
// Completed
```

### With empty source observable

```go
obs := ro.Pipe[string, string](
    ro.Empty[string](),
    ro.StartWith("prefix1", "prefix2"),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: prefix1
// Next: prefix2
// Completed
```

### With no prefix values

```go
obs := ro.Pipe[string, string](
    ro.Just("a", "b"),
    ro.StartWith[string](),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: a
// Next: b
// Completed
```

### With time-based observable

```go
obs := ro.Pipe[int64, int64](
    ro.Interval(100*time.Millisecond),
    ro.StartWith(int64(-1)),
    ro.Take(3),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
time.Sleep(400 * time.Millisecond)
sub.Unsubscribe()

// Next: -1
// Next: 0
// Next: 1
// Next: 2
// Completed
```