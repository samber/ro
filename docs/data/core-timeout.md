---
name: Timeout
slug: timeout
sourceRef: operator_utility.go#L419
type: core
category: utility
signatures:
  - "func Timeout[T any](duration time.Duration)"
playUrl:
variantHelpers:
  - core#utility#timeout
similarHelpers:
  - core#utility#delay
  - core#utility#sampletime
  - core#utility#throttletime
position: 90
---

Raises an error if the source Observable does not emit any item within the specified duration. The timeout resets after each emission.

```go
obs := ro.Pipe[int64, int64](
    ro.Interval(200*time.Millisecond),
    ro.Timeout(100*time.Millisecond),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
time.Sleep(300 * time.Millisecond)
sub.Unsubscribe()

// Error: timeout after 100ms
```

### With fast emissions (no timeout)

```go
obs := ro.Pipe[int64, int64](
    ro.Interval(50*time.Millisecond),
    ro.Timeout(200*time.Millisecond),
    ro.Take(3),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
time.Sleep(300 * time.Millisecond)
sub.Unsubscribe()

// Next: 0
// Next: 1
// Next: 2
// Completed
```

### With slow emissions (timeout occurs)

```go
obs := ro.Pipe[int64, int64](
    ro.Interval(500*time.Millisecond),
    ro.Timeout(200*time.Millisecond),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
time.Sleep(800 * time.Millisecond)
sub.Unsubscribe()

// Error: timeout after 200ms
```

### With delayed first emission

```go
obs := ro.Pipe[string, string](
    ro.Pipe[string, string](
        ro.Just("delayed"),
        ro.Delay(300*time.Millisecond),
    ),
    ro.Timeout(100*time.Millisecond),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(500 * time.Millisecond)
sub.Unsubscribe()

// Error: timeout after 100ms (before first emission)
```

### With error in source (propagates immediately)

```go
obs := ro.Pipe[string, string](
    ro.Pipe[string, string](
        ro.Just("will error"),
        ro.Throw[string](errors.New("source error")),
    ),
    ro.Timeout(1*time.Second),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Error: source error (propagates before timeout)
```

### With multiple emissions and varying intervals

```go
obs := ro.Pipe[string, string](
    ro.Pipe[string, string](
        ro.Just("fast"),
        ro.Delay(50*time.Millisecond),
    ),
    ro.Pipe[string, string](
        ro.Just("slow"),
        ro.Delay(300*time.Millisecond),
    ),
    ro.Timeout(200*time.Millisecond),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(400 * time.Millisecond)
sub.Unsubscribe()

// Next: fast (emitted within timeout)
// Error: timeout after 200ms (waiting for slow)
```