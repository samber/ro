---
name: Throw
slug: throw
sourceRef: operator_creation.go#L46
type: core
category: creation
signatures:
  - "func Throw[T any](err error)"
playUrl: https://go.dev/play/p/YuaGv9G-YIf
variantHelpers:
  - core#creation#throw
similarHelpers:
  - core#creation#empty
  - core#creation#never
position: 33
---

Creates an Observable that emits no items and immediately terminates with an error. This creation operator is very useful for unit tests.

```go
obs := ro.Throw[int](errors.New("something went wrong"))

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Error: something went wrong
```

### With custom error types

```go
type CustomError struct {
    Code    int
    Message string
}

func (e *CustomError) Error() string {
    return fmt.Sprintf("Error %d: %s", e.Code, e.Message)
}

obs := ro.Throw[string](&CustomError{Code: 404, Message: "Not found"})

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Error: Error 404: Not found
```

### For unit tests

```go
// pipeline.go
var pipeline ro.PipeOp(
    ro.Map(func(x int) int {
        return x*2
    })
    ro.Catch(func(err error) ro.Observable[int] {
        return ro.Just(42)
    }),
)

// pipeline_test.go
func TestMyPipeline(t *testing.T) {
    // testing empty source
    obs := pipeline(ro.Empty[int]())

    values, err := ro.Collect(obs)
    defer sub.Unsubscribe()

    t.Assert(...)

    // testing broken source
    obs := pipeline(ro.Throw[int](errors.New("something went wrong")))

    values, err = ro.Collect(obs)
    defer sub.Unsubscribe()

    t.Assert(...)

    // testing inactive stream
    obs := pipeline(ro.Never[int]())

    values, err = ro.Collect(obs)
    defer sub.Unsubscribe()

    t.Assert(...)
}
```

### With Retry

```go
obs := ro.Pipe[int, int](
    ro.Throw[int](errors.New("network error")),
    ro.Retry(3),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
time.Sleep(100 * time.Millisecond) // Give retry attempts time
sub.Unsubscribe()

// Will attempt retry 3 times before propagating the error
// Error: network error
```

### With error handling

```go
obs := ro.Pipe[int, int](
    ro.Throw[int](errors.New("original error")),
    ro.Catch(func(err error) ro.Observable[int] {
        return ro.Just(42) // fallback value
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 42 (fallback from Catch)
// Completed
```
