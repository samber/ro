---
name: SequenceEqual
slug: sequenceequal
sourceRef: operator_conditional.go#L222
type: core
category: conditional
signatures:
  - "func SequenceEqual[T comparable](compareTo Observable[T])"
playUrl: https://go.dev/play/p/cBIQlH01byQ
variantHelpers:
  - core#conditional#sequenceequal
similarHelpers: []
position: 50
---

Determines whether two observable sequences emit the same sequence of values.

```go
source := ro.Just(1, 2, 3)
compareTo := ro.Just(1, 2, 3)

obs := ro.Pipe[int, bool](
    source,
    ro.SequenceEqual(compareTo),
)

sub := obs.Subscribe(ro.PrintObserver[bool]())
defer sub.Unsubscribe()

// Next: true
// Completed
```

### With different sequences

```go
source := ro.Just(1, 2, 3)
compareTo := ro.Just(1, 2, 4)

obs := ro.Pipe[int, bool](
    source,
    ro.SequenceEqual(compareTo),
)

sub := obs.Subscribe(ro.PrintObserver[bool]())
defer sub.Unsubscribe()

// Next: false
// Completed
```

### With different length sequences

```go
source := ro.Just(1, 2, 3)
compareTo := ro.Just(1, 2)

obs := ro.Pipe[int, bool](
    source,
    ro.SequenceEqual(compareTo),
)

sub := obs.Subscribe(ro.PrintObserver[bool]())
defer sub.Unsubscribe()

// Next: false
// Completed
```

### With strings

```go
source := ro.Just("hello", "world")
compareTo := ro.Just("hello", "world")

obs := ro.Pipe[string, bool](
    source,
    ro.SequenceEqual(compareTo),
)

sub := obs.Subscribe(ro.PrintObserver[bool]())
defer sub.Unsubscribe()

// Next: true
// Completed
```