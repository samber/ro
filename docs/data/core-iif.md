---
name: Iif
slug: iif
sourceRef: operator_conditional.go#L176
type: core
category: conditional
signatures:
  - "func Iif[T any](condition func() bool, trueSource Observable[T], falseSource Observable[T])"
playUrl: https://go.dev/play/p/t-sNgL5EZA-
variantHelpers:
  - core#conditional#iif
similarHelpers: []
position: 30
---

Conditionally selects between two observables based on a condition function.

```go
condition := func() bool {
    return time.Now().Hour() < 12 // Before noon
}

obs := ro.Iif(
    condition,
    ro.Just("Good morning"),
    ro.Just("Good afternoon"),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: Good morning (if before noon) or Good afternoon (if after noon)
// Completed
```

### Dynamic condition based on data

```go
useFallback := false
condition := func() bool {
    return !useFallback
}

primarySource := ro.Just(1, 2, 3)
fallbackSource := ro.Just(0)

obs := ro.Iif(condition, primarySource, fallbackSource)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Completed
```

### With error handling

```go
isValid := true
condition := func() bool {
    return isValid
}

validSource := ro.Just("success")
errorSource := ro.Throw[string](errors.New("invalid state"))

obs := ro.Iif(condition, validSource, errorSource)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: success
// Completed
```