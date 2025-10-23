---
name: ElementAt
slug: elementat
sourceRef: operator_filter.go#L682
type: core
category: filtering
signatures:
  - "func ElementAt[T any](nth int)"
playUrl:
variantHelpers:
  - core#filtering#elementat
similarHelpers:
  - core#filtering#elementatordefault
  - core#filtering#first
  - core#filtering#last
  - core#filtering#take
position: 70
---

Emits only the nth item emitted by an Observable. If the source Observable emits fewer than n items, ElementAt will emit an error.

```go
obs := ro.Pipe[string, string](
    ro.Just("apple", "banana", "cherry", "date"),
    ro.ElementAt(2),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: cherry
// Completed
```

### With zero-based indexing

```go
obs := ro.Pipe[string, string](
    ro.Just("first", "second", "third"),
    ro.ElementAt(0),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: first
// Completed
```

### With out of bounds index

```go
obs := ro.Pipe[string, string](
    ro.Just("a", "b", "c"),
    ro.ElementAt(5),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Error: element at index 5 not found
```

### With numbers

```go
obs := ro.Pipe[int, int](
    ro.Just(10, 20, 30, 40, 50),
    ro.ElementAt(3),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 40
// Completed
```

### With time-based observable

```go
obs := ro.Pipe[int64, int64](
    ro.Interval(100*time.Millisecond),
    ro.ElementAt(5),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
time.Sleep(800 * time.Millisecond)
sub.Unsubscribe()

// Next: 5 (emitted after 500ms)
// Completed
```