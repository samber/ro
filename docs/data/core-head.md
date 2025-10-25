---
name: Head
slug: head
sourceRef: operator_filter.go#L509
type: core
category: filtering
signatures:
  - "func Head[T any]()"
playUrl: https://go.dev/play/p/TmhTvpuKAp_U
variantHelpers:
  - core#filtering#head
similarHelpers:
  - core#filtering#first
  - core#filtering#tail
  - core#filtering#take
position: 50
---

Emits only the first item emitted by an Observable. If the source Observable is empty, Head will emit an error.

```go
obs := ro.Pipe[string, string](
    ro.Just("first", "second", "third"),
    ro.Head(),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: first
// Completed
```

### With empty observable

```go
obs := ro.Pipe[string, string](
    ro.Empty[string](),
    ro.Head(),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Error: head of empty observable
```

### With single item

```go
obs := ro.Pipe[string, string](
    ro.Just("only one"),
    ro.Head(),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: only one
// Completed
```

### With numbers

```go
obs := ro.Pipe[int, int](
    ro.Just(10, 20, 30, 40, 50),
    ro.Head(),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 10
// Completed
```

### With time-based emissions

```go
obs := ro.Pipe[int64, int64](
    ro.Interval(100*time.Millisecond),
    ro.Head(),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
time.Sleep(200 * time.Millisecond)
sub.Unsubscribe()

// Next: 0 (emitted immediately)
// Completed
```

### With error in source

```go
obs := ro.Pipe[string, string](
    ro.Throw[string](errors.New("source error")),
    ro.Head(),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Error: source error
```