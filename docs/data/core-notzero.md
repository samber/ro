---
name: NotZero
slug: notzero
sourceRef: operator_filter.go#L141
type: core
category: filtering
signatures:
  - "func NotZero[T comparable]()"
playUrl:
variantHelpers:
  - core#filtering#notzero
similarHelpers:
  - core#filtering#filter
position: 65
---

Emits only the items that are not the zero value of their type (`0`, `""`, `false`, `nil`, empty struct...).

```go
obs := ro.Pipe[int, int](
    ro.Just(0, 1, 0, 2, 0, 3),
    ro.NotZero[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Completed
```

### With strings

```go
obs := ro.Pipe[string, string](
    ro.Just("", "apple", "", "banana"),
    ro.NotZero[string](),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: apple
// Next: banana
// Completed
```
