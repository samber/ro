---
name: Race
slug: race
sourceRef: operator_creation.go#L550
type: core
category: combining
signatures:
  - "func Race[T any](sources ...Observable[T])"
  - "func Amb[T any](sources ...Observable[T])"
playUrl:
variantHelpers:
  - core#creation#race
  - core#creation#amb
similarHelpers:
  - core#creation#merge
  - core#creation#combinelatestx
position: 43
---

Creates an Observable that mirrors the first source Observable to emit an item or send a notification. Amb is an alias for Race.

```go
fast := ro.Pipe[int64, string](
    ro.Timer(100*time.Millisecond),
    ro.Map(func(_ int64) string { return "fast" }),
)
slow := ro.Pipe[int64, string](
    ro.Timer(200*time.Millisecond),
    ro.Map(func(_ int64) string { return "slow" }),
)

obs := ro.Race(fast, slow)

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(300 * time.Millisecond)
sub.Unsubscribe()

// Next: "fast" (from the faster observable)
// Completed
```

### With multiple sources

```go
sources := []Observable[int]{
    ro.Pipe[int64, int](ro.Timer(300*time.Millisecond), ro.Map(func(_ int64) int { return 1 })),
    ro.Pipe[int64, int](ro.Timer(100*time.Millisecond), ro.Map(func(_ int64) int { return 2 })),
    ro.Pipe[int64, int](ro.Timer(200*time.Millisecond), ro.Map(func(_ int64) int { return 3 })),
}

obs := ro.Race(sources...)

sub := obs.Subscribe(ro.PrintObserver[int]())
time.Sleep(400 * time.Millisecond)
sub.Unsubscribe()

// Next: 2 (from the 100ms timer)
// Completed
```

### With error handling

```go
success := ro.Pipe[int64, int](ro.Timer(200*time.Millisecond), ro.Map(func(_ int64) int { return 42 }))
failure := ro.Pipe[int64, int](ro.Timer(100*time.Millisecond), ro.Throw[int](errors.New("failed")))

obs := Race(success, failure)

sub := obs.Subscribe(ro.PrintObserver[int]())
time.Sleep(300 * time.Millisecond)
sub.Unsubscribe()

// Error: failed
```

### Timeout pattern

```go
data := ro.Future[int](func(resolve func(int), reject func(error)) {
    go func() {
        time.Sleep(2 * time.Second)
        resolve(42)
    }()
})

timeout := ro.Timer(1 * time.Second)
result := Race(data, timeout)

sub := result.Subscribe(ro.PrintObserver[any]())
time.Sleep(2500 * time.Millisecond)
sub.Unsubscribe()

// Next: 0 (from timeout after 1 second)
// Completed
```

### Fallback pattern

```go
primary := ro.Future[string](func(resolve func(string), reject func(error)) {
    go func() {
        time.Sleep(500 * time.Millisecond)
        resolve("primary result")
    }()
})

fallback := ro.Just("fallback value")
result := Race(primary, fallback)

sub := result.Subscribe(ro.PrintObserver[string]())
time.Sleep(1000 * time.Millisecond)
sub.Unsubscribe()

// Next: "fallback value"
// Completed
```