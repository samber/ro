---
name: RandIntN
slug: randintn
sourceRef: operator_creation.go#L118
type: core
category: creation
signatures:
  - "func RandIntN(n int64)"
playUrl:
variantHelpers:
  - core#creation#randintn
similarHelpers:
  - core#creation#randfloat64
position: 41
---

Creates an Observable that emits a single random integer between 0 (inclusive) and n (exclusive).

```go
obs := ro.RandIntN(100)

sub := obs.Subscribe(ro.PrintObserver[int64]())
defer sub.Unsubscribe()

// Next: <random number between 0-99>
// Completed
```

### Multiple random numbers

```go
obs := ro.Pipe[int64, int64](
    ro.RandIntN(1000),
    ro.RepeatWithInterval(ro.RandIntN(1000), 100*time.Millisecond),
    ro.Take[int64](5),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
time.Sleep(1000 * time.Millisecond)
sub.Unsubscribe()

// Next: <random number 0-999> (immediately)
// Next: <random number 0-999> (after 100ms)
// Next: <random number 0-999> (after 200ms)
// Next: <random number 0-999> (after 300ms)
// Next: <random number 0-999> (after 400ms)
// Completed
```

### With different ranges

```go
// D6 dice roll
diceObs := ro.Pipe[int64, int64](
    ro.RandIntN(6), // 0-5, so add 1 for 1-6
    ro.Map(func(n int64) int64 { return n + 1 }),
)

sub := diceObs.Subscribe(ro.PrintObserver[int64]())
defer sub.Unsubscribe()

// Next: <random number 1-6>
// Completed
```

### For simulation or testing

```go
// Simulate random delays
delays := ro.Pipe[int64, time.Duration](
    ro.RandIntN(5000), // 0-4999ms
    ro.Map(func(ms int64) time.Duration {
        return time.Duration(ms) * time.Millisecond
    }),
)

sub := delays.Subscribe(ro.PrintObserver[time.Duration]())
defer sub.Unsubscribe()

// Next: <random duration between 0-4999ms>
// Completed
```

### With error probability

```go
shouldError := ro.Pipe[int64, bool](
    ro.RandIntN(10), // 0-9
    ro.Map(func(n int64) bool {
        return n == 0 // 10% chance of error
    }),
)

sub := shouldError.Subscribe(ro.PrintObserver[bool]())
defer sub.Unsubscribe()

// Next: true (10% chance) or false (90% chance)
// Completed
```