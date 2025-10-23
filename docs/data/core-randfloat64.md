---
name: RandFloat64
slug: randfloat64
sourceRef: operator_creation.go#L122
type: core
category: creation
signatures:
  - "func RandFloat64()"
playUrl:
variantHelpers:
  - core#creation#randfloat64
similarHelpers:
  - core#creation#randintn
position: 42
---

Creates an Observable that emits a single random float64 value between 0.0 (inclusive) and 1.0 (exclusive).

```go
obs := ro.RandFloat64()

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: <random float between 0.0-1.0>
// Completed
```

### Multiple random values

```go
obs := ro.Pipe[float64, float64](
    ro.RandFloat64(),
    RepeatWithInterval(ro.RandFloat64(), 100*time.Millisecond),
    ro.Take[float64](5),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
time.Sleep(1000 * time.Millisecond)
sub.Unsubscribe()

// Next: <random float 0.0-1.0> (immediately)
// Next: <random float 0.0-1.0> (after 100ms)
// Next: <random float 0.0-1.0> (after 200ms)
// Next: <random float 0.0-1.0> (after 300ms)
// Next: <random float 0.0-1.0> (after 400ms)
// Completed
```

### For probability calculations

```go
// 75% chance of success
successChance := ro.Pipe[float64, bool](
    ro.RandFloat64(),
    ro.Map(func(f float64) bool {
        return f < 0.75
    }),
)

sub := successChance.Subscribe(ro.PrintObserver[bool]())
defer sub.Unsubscribe()

// Next: true (75% chance) or false (25% chance)
// Completed
```

### Custom range mapping

```go
// Random temperature between 15.0 and 25.0 degrees
temperature := ro.Pipe[float64, float64](
    ro.RandFloat64(),
    ro.Map(func(f float64) float64 {
        return 15.0 + (f * 10.0) // Map 0-1 to 15-25
    }),
)

sub := temperature.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: <random float between 15.0-25.0>
// Completed
```

### For testing variations

```go
// Simulate network latency with random jitter
baseLatency := 100 * time.Millisecond
jitterRange := 50 * time.Millisecond

latency := ro.Pipe[float64, time.Duration](
    ro.RandFloat64(),
    ro.Map(func(f float64) time.Duration {
        jitter := time.Duration(f * float64(jitterRange))
        return baseLatency + jitter
    }),
)

sub := latency.Subscribe(ro.PrintObserver[time.Duration]())
defer sub.Unsubscribe()

// Next: <duration between 100-150ms>
// Completed
```