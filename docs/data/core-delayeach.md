---
name: DelayEach
slug: delayeach
sourceRef: operator_utility.go#L371
type: core
category: utility
signatures:
  - "func DelayEach[T any](duration time.Duration)"
playUrl: https://go.dev/play/p/a9opbDTetAz
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
values, err := ro.Collect(
    ro.Pipe[string, string](
        ro.Just("A", "B", "C"),
        ro.DelayEach[string](1*time.Millisecond),
    ),
)
for _, v := range values {
    fmt.Printf("Next: %s\n", v)
}
if err != nil {
    fmt.Printf("Error: %v\n", err)
} else {
    fmt.Println("Completed")
}

// Next: A
// Next: B
// Next: C
// Completed
```
