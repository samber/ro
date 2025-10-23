---
name: IntervalWithInitial
slug: intervalwithinitial
sourceRef: operator_creation.go#L80
type: core
category: creation
signatures:
  - "func IntervalWithInitial(initial time.Duration, interval time.Duration)"
playUrl:
variantHelpers:
  - core#creation#intervalwithinitial
similarHelpers:
  - core#creation#interval
  - core#creation#timer
  - core#creation#range
position: 21
---

Creates an Observable that emits sequential numbers starting after an initial delay, then continuing at specified intervals.

```go
obs := ro.IntervalWithInitial(200*time.Millisecond, 100*time.Millisecond)

sub := obs.Subscribe(ro.PrintObserver[int64]())
time.Sleep(650 * time.Millisecond)
sub.Unsubscribe()

// Next: 0 (after 200ms - initial delay)
// Next: 1 (after 300ms)
// Next: 2 (after 400ms)
// Next: 3 (after 500ms)
// Next: 4 (after 600ms)
```

### Long initial delay with short interval

```go
obs := ro.IntervalWithInitial(1*time.Second, 200*time.Millisecond)

sub := obs.Subscribe(ro.PrintObserver[int64]())
time.Sleep(2000 * time.Millisecond)
sub.Unsubscribe()

// Next: 0 (after 1000ms - initial delay)
// Next: 1 (after 1200ms)
// Next: 2 (after 1400ms)
// Next: 3 (after 1600ms)
// Next: 4 (after 1800ms)
// Next: 5 (after 2000ms)
```

### Short initial delay with long interval

```go
obs := ro.IntervalWithInitial(100*time.Millisecond, 1*time.Second)

sub := obs.Subscribe(ro.PrintObserver[int64]())
time.Sleep(2500 * time.Millisecond)
sub.Unsubscribe()

// Next: 0 (after 100ms - initial delay)
// Next: 1 (after 1100ms)
// Next: 2 (after 2100ms)
```

### Practical example: Delayed heartbeat

```go
heartbeat := ro.IntervalWithInitial(1*time.Second, 500*time.Millisecond)

sub := heartbeat.Subscribe(ro.PrintObserver[int64]())
time.Sleep(3000 * time.Millisecond)
sub.Unsubscribe()

// Next: 0 (after 1000ms - initial delay before heartbeat starts)
// Next: 1 (after 1500ms)
// Next: 2 (after 2000ms)
// Next: 3 (after 2500ms)
// Next: 4 (after 3000ms)
```

### With Take for limited emissions

```go
obs := ro.Pipe[int64, int64](
    ro.IntervalWithInitial(200*time.Millisecond, 100*time.Millisecond),
    ro.Take[int64](5),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
defer sub.Unsubscribe()

// Next: 0 (after 200ms)
// Next: 1 (after 300ms)
// Next: 2 (after 400ms)
// Next: 3 (after 500ms)
// Next: 4 (after 600ms)
// Completed
```

### Edge case: Zero initial delay

```go
obs := ro.IntervalWithInitial(0, 100*time.Millisecond)

sub := obs.Subscribe(ro.PrintObserver[int64]())
time.Sleep(550 * time.Millisecond)
sub.Unsubscribe()

// Behaves like regular Interval
// Next: 0 (immediately)
// Next: 1 (after 100ms)
// Next: 2 (after 200ms)
// Next: 3 (after 300ms)
// Next: 4 (after 400ms)
// Next: 5 (after 500ms)
```

### Edge case: Very long initial delay

```go
obs := ro.IntervalWithInitial(5*time.Second, 1*time.Second)

sub := obs.Subscribe(ro.PrintObserver[int64]())
time.Sleep(8000 * time.Millisecond)
sub.Unsubscribe()

// Next: 0 (after 5000ms - long initial delay)
// Next: 1 (after 6000ms)
// Next: 2 (after 7000ms)
```