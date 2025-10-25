---
name: IgnoreElements
slug: ignoreelements
sourceRef: operator_filter.go#L101
type: core
category: filtering
signatures:
  - "func IgnoreElements[T any]()"
playUrl: https://go.dev/play/p/glDG6E-gZ1V
variantHelpers:
  - core#filtering#ignoreelements
similarHelpers: []
position: 70
---

Ignores all elements emitted by the source observable, only allowing completion or error notifications to pass through.

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.IgnoreElements[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Completed (no values emitted)
```

### With error handling

```go
obs := ro.Pipe[int, int](
    ro.Concat(
        ro.Just(1, 2, 3),
        ro.Throw[int](errors.New("something went wrong")),
    ),
    ro.IgnoreElements[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Error: something went wrong
```

### For side effects only

```go
// Use IgnoreElements when you only care about side effects
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.Tap(func(n int) {
        fmt.Printf("Processing item: %d\n", n)
    }),
    ro.IgnoreElements[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Processing item: 1
// Processing item: 2
// Processing item: 3
// Processing item: 4
// Processing item: 5
// Completed
```

### With delayed completion

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.Delay(100*time.Millisecond),
    ro.IgnoreElements[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Completed after 100ms (no values emitted)
```