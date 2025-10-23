---
name: ElementAtOrDefault
slug: elementatordefault
sourceRef: operator_filter.go#L717
type: core
category: filtering
signatures:
  - "func ElementAtOrDefault[T any](nth int64, fallback T)"
playUrl:
variantHelpers:
  - core#filtering#elementatordefault
similarHelpers:
  - core#filtering#elementat
  - core#filtering#first
  - core#filtering#last
  - core#filtering#take
position: 71
---

Emits only the nth item emitted by an Observable. If the source Observable emits fewer than n items, ElementAtOrDefault will emit a fallback value.

```go
obs := ro.Pipe[string, string](
    ro.Just("a", "b", "c"),
    ro.ElementAtOrDefault(5, "fallback"),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: fallback
// Completed
```

### Within bounds (emits the nth item)

```go
obs := ro.Pipe[string, string](
    ro.Just("first", "second", "third"),
    ro.ElementAtOrDefault(1, "fallback"),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: second
// Completed
```

### With zero-based indexing and fallback

```go
obs := ro.Pipe[string, string](
    ro.Just("apple"),
    ro.ElementAtOrDefault(0, "no fruit"),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: apple
// Completed
```

### With numbers and fallback

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.ElementAtOrDefault(10, 999),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 999
// Completed
```

### With empty observable

```go
obs := ro.Pipe[string, string](
    ro.Empty[string](),
    ro.ElementAtOrDefault(0, "default value"),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: default value
// Completed
```

### With negative index (will panic)

```go
// This will panic because nth cannot be negative
defer func() {
    if r := recover(); r != nil {
        fmt.Println("Recovered from panic:", r)
    }
}()

obs := ro.Pipe[string, string](
    ro.Just("a", "b"),
    ro.ElementAtOrDefault(-1, "fallback"), // This will panic
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()
```
