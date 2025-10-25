---
name: OnErrorResumeNextWith
slug: onerrorresumenextwith
sourceRef: operator_error_handling.go#L53
type: core
category: error-handling
signatures:
  - "func OnErrorResumeNextWith[T any](finally ...Observable[T])"
playUrl: https://go.dev/play/p/9XLTAOginbK
variantHelpers:
  - core#error-handling#onerrorresumenextwith
similarHelpers:
  - core#error-handling#catch
  - core#error-handling#onerrorreturn
position: 30
---

Begins emitting a second observable sequence if it encounters an error with the first observable.

```go
primary := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.MapErr(func(i int) (int, error) {
        if i == 3 {
            return 0, errors.New("error occurred")
        }
        return i, nil
    }),
)

fallback := ro.Just(99, 100, 101)

obs := ro.Pipe[int, int](primary, ro.OnErrorResumeNextWith(fallback))

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 99 (fallback starts)
// Next: 100
// Next: 101
// Completed
```

### With multiple fallback sequences

```go
primary := ro.Pipe[string, string](
    ro.Just("data1", "data2"),
    ro.Throw[string](errors.New("primary failed")),
)

fallback1 := ro.Just("fallback1", "fallback2")
fallback2 := ro.Just("final1", "final2")

obs := ro.Pipe[string, string](primary, ro.OnErrorResumeNextWith(fallback1, fallback2))

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: "data1"
// Next: "data2"
// Next: "fallback1"
// Next: "fallback2"
// (fallback2 is ignored, only first fallback is used)
```

### With empty fallback

```go
primary := ro.Throw[int](errors.New("always fails"))
fallback := ro.Empty[int]()

obs := ro.Pipe[int, int](primary, ro.OnErrorResumeNextWith(fallback))

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Completed (no items, no error)
```

### API fallback pattern

```go
primaryAPI := func() Observable[string] {
    return ro.Pipe[string, string](
        ro.Just("user_data"),
        ro.Throw[string](errors.New("API timeout")),
    )
}

cacheAPI := func() Observable[string] {
    return ro.Just("cached_data")
}

obs := ro.Pipe[string, string](
    primaryAPI(),
    ro.OnErrorResumeNextWith(cacheAPI()),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: "user_data" (from primary before error)
// Next: "cached_data" (from fallback)
// Completed
```

### Database connection fallback

```go
connectPrimary := func() Observable[string] {
    // Simulate primary database failure
    return ro.Throw[string](errors.New("primary database unavailable"))
}

connectSecondary := func() Observable[string] {
    return ro.Just("connected to secondary database")
}

obs := ro.Pipe[string, string](
    connectPrimary(),
    ro.OnErrorResumeNextWith(connectSecondary()),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: "connected to secondary database"
// Completed
```

### With conditional fallback

```go
shouldUseFallback := true
primary := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.MapErr(func(i int) (int, error) {
        if i == 3 && shouldUseFallback {
            return 0, errors.New("switch to fallback")
        }
        return i, nil
    }),
)

fallback := ro.Pipe[int, int](
    ro.Just(4, 5, 6),
    ro.Map(func(i int) int {
        return i * 10
    }),
)

obs := ro.Pipe[int, int](primary, ro.OnErrorResumeNextWith(fallback))

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 40 (fallback: 4*10)
// Next: 50 (fallback: 5*10)
// Next: 60 (fallback: 6*10)
// Completed
```