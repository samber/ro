---
name: DelayEach
slug: delayeach
sourceRef: operator_utility.go#L371
type: core
category: utility
signatures:
  - "func DelayEach[T any](duration time.Duration)"
playUrl: ""
variantHelpers:
  - core#utility#delayeach
similarHelpers:
  - core#utility#delay
  - core#utility#timeout
position: 221
---

Delays each item emitted by the source Observable by a fixed duration before forwarding it.

Unlike Delay which shifts all emissions by the same amount, DelayEach introduces a per-item pause.

```go
obs := ro.Pipe[string, string](
    ro.Just("A", "B", "C"),
    ro.DelayEach[string](100 * time.Millisecond),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// (100ms pause)
// Next: A
// (100ms pause)
// Next: B
// (100ms pause)
// Next: C
// Completed
```
