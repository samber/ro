---
name: Amb
slug: amb
sourceRef: operator_creation.go#L559
type: core
category: combining
signatures:
  - "func Amb[T any](sources ...Observable[T])"
  - "func Race[T any](sources ...Observable[T])"
playUrl: https://go.dev/play/p/-YvhnpQFVNS
variantHelpers:
  - core#combining#amb
  - core#creation#race
similarHelpers:
  - core#combining#race
  - core#combining#racewith
position: 20
---

Creates an Observable that mirrors the first source Observable to emit a next, error or complete notification. It's an alias for Race.

The Observable cancels subscriptions to all other Observables once one emits. It completes when the winning source Observable completes.

```go
obs1 := ro.Pipe[string, string](
    ro.Just("fast"),
    ro.Delay(100*time.Millisecond),
)

obs2 := ro.Pipe[string, string](
    ro.Just("slow"),
    ro.Delay(200*time.Millisecond),
)

obs3 := ro.Pipe[string, string](
    ro.Just("slowest"),
    ro.Delay(300*time.Millisecond),
)

ambObs := ro.Amb(obs1, obs2, obs3)

sub := ambObs.Subscribe(ro.PrintObserver[string]())
time.Sleep(400 * time.Millisecond)
sub.Unsubscribe()

// Next: fast
// Completed
```

### With immediate winner

```go
instant := ro.Just("immediate")
delayed := ro.Pipe[string, string](
    ro.Just("delayed"),
    ro.Delay(100*time.Millisecond),
)

ambObs := ro.Amb(delayed, instant)

sub := ambObs.Subscribe(ro.PrintObserver[string]())
time.Sleep(200 * time.Millisecond)
sub.Unsubscribe()

// Next: immediate
// Completed
```

### With error propagation

```go
obs1 := ro.Pipe[string, string](
    ro.Just("success"),
    ro.Delay(100*time.Millisecond),
)

obs2 := ro.Throw[string](errors.New("failed"))

ambObs := ro.Amb(obs1, obs2)

sub := ambObs.Subscribe(ro.PrintObserver[string]())
time.Sleep(200 * time.Millisecond)
sub.Unsubscribe()

// Next: success
// Completed
```

### With empty sources

```go
ambObs := ro.Amb[string]() // No sources provided

sub := ambObs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Completed (empty observable)
```