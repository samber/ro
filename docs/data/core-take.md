---
name: Take
slug: take
sourceRef: operator_filter.go#L299
type: core
category: filtering
signatures:
  - "func Take[T any](count int64)"
playUrl: https://go.dev/play/p/IC_hJMsg7yk
variantHelpers:
  - core#filtering#take
similarHelpers:
  - core#filtering#takewhile
  - core#filtering#takewhilewithcontext
  - core#filtering#takewhilei
  - core#filtering#takewhileiwithcontext
  - core#filtering#takelast
  - core#filtering#takeuntil
  - core#filtering#head
position: 10
---

Emits only the first n items emitted by an Observable. If the count is greater than the number of items emitted by the source Observable, Take will emit all items. If the count is zero, Take will not emit any items.

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.Take[int](3),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Completed
```

### Edge case: Taking more items than available

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.Take[int](5),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Completed
```

### Edge case: Taking zero items

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.Take[int](0),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Completed
```