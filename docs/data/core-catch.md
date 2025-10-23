---
name: Catch
slug: catch
sourceRef: operator_error_handling.go#L25
type: core
category: error-handling
signatures:
  - "func Catch[T any](finally func(err error) Observable[T])"
playUrl:
variantHelpers:
  - core#error-handling#catch
similarHelpers:
  - core#error-handling#onerrorresumenextwith
  - core#error-handling#onerrorreturn
position: 0
---

Catches errors on the observable to be handled by returning a new observable.

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.MapErr(func(i int) (int, error) {
        if i == 3 {
            return 0, errors.New("number 3 is not allowed")
        }
        return i * 2, nil
    }),
    ro.Catch(func(err error) ro.Observable[int] {
        fmt.Printf("Error: %v\n", err)
        return ro.Just(99) // Fallback value
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 2 (1*2)
// Next: 4 (2*2)
// Error: number 3 is not allowed
// Next: 99 (fallback value)
// Completed
```

### With retry logic

```go
attempt := 0
obs := ro.Pipe[int, int](
    ro.Defer(func() ro.Observable[int] {
        attempt++
        if attempt <= 2 {
            return ro.Pipe[int, int](
                ro.Just(1),
                ro.Throw[int](errors.New("network error")),
            )
        }
        return ro.Just(42)
    }),
    ro.Catch(func(err error) ro.Observable[int] {
        fmt.Printf("Attempt %d failed: %v\n", attempt, err)
        if attempt < 3 {
            return ro.Empty[int]() // Stop this attempt, allow retry
        }
        return ro.Just(-1) // Final fallback
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Attempt 1 failed: network error
// Attempt 2 failed: network error
// Next: 42 (success on 3rd attempt)
// Completed
```

### With different error types

```go
obs := ro.Pipe[string, string](
    ro.Just("data1", "data2", "invalid"),
    ro.MapErr(func(s string) (string, error) {
        if s == "invalid" {
            return "", errors.New("invalid data")
        }
        return strings.ToUpper(s), nil
    }),
    ro.Catch(func(err error) ro.Observable[string] {
        if strings.Contains(err.Error(), "invalid") {
            return ro.Just("DEFAULT") // Handle validation errors
        }
        return ro.Throw[string](err) // Re-throw other errors
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: "DATA1"
// Next: "DATA2"
// Next: "DEFAULT"
// Completed
```

### With logging and fallback sequence

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4),
    ro.MapErr(func(i int) (int, error) {
        if i%2 == 0 {
            return 0, fmt.Errorf("even number %d rejected", i)
        }
        return i, nil
    }),
    ro.Catch(func(err error) ro.Observable[int] {
        log.Printf("Error caught: %v", err)
        // Provide fallback sequence
        return ro.FromSlice([]int{100, 200, 300})
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Error caught: even number 2 rejected
// Next: 100
// Next: 200
// Next: 300
// Completed
```