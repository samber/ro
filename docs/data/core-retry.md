---
name: Retry
slug: retry
sourceRef: operator_error_handling.go#L131
type: core
category: error-handling
signatures:
  - "func Retry[T any]()"
  - "func RetryWithConfig[T any](opts RetryConfig)"
playUrl:
variantHelpers:
  - core#error-handling#retry
  - core#error-handling#retrywithconfig
similarHelpers: []
position: 10
---

Retries the source observable sequence when it encounters an error. Retry uses infinite retries with default settings, while RetryWithConfig provides configurable retry behavior.

```go
attempt := 0
obs := ro.Pipe[string, string](
    ro.Defer(func() Observable[string] {
        attempt++
        if attempt < 3 {
            return ro.Pipe[string, string](
                ro.Just("data"),
                ro.Throw[string](errors.New("temporary failure")),
            )
        }
        return ro.Just("success!")
    }),
    ro.Retry[string](),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(100 * time.Millisecond) // Allow time for retries
sub.Unsubscribe()

// Next: "success!" (after 3 attempts)
// Completed
```

### RetryWithConfig with limited retries

```go
attempt := 0
obs := ro.Pipe[int, int](
    ro.Defer(func() Observable[int] {
        attempt++
        if attempt == 1 {
            return ro.Throw[int](errors.New("first attempt failed"))
        }
        return ro.Just(42)
    }),
    ro.RetryWithConfig[int](RetryConfig{
        MaxRetries:     3,
        Delay:          100 * time.Millisecond,
        ResetOnSuccess: true,
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
time.Sleep(500 * time.Millisecond)
sub.Unsubscribe()

// Next: 42 (success on second attempt)
// Completed
```

### With exponential backoff

```go
obs := ro.Pipe[string, string](
    ro.Defer(func() Observable[string] {
        return ro.Pipe[string, string](
            ro.Just("api_data"),
            ro.Throw[string](errors.New("rate limited")),
        )
    }),
    ro.RetryWithConfig[string](RetryConfig{
        MaxRetries:     5,
        Delay:          1 * time.Second,
        ResetOnSuccess: true,
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(6 * time.Second)
sub.Unsubscribe()

// Would retry up to 5 times with 1-second delays
// (assuming API continues to fail)
```

### With ResetOnSuccess behavior

```go
successCount := 0
obs := ro.Pipe[int, int](
    ro.Defer(func() Observable[int] {
        successCount++
        if successCount <= 2 {
            return ro.Just(successCount)
        }
        return ro.Throw[int](errors.New("suddenly failed"))
    }),
    ro.RetryWithConfig[int](RetryConfig{
        MaxRetries:     3,
        Delay:          50 * time.Millisecond,
        ResetOnSuccess: true, // Success resets retry counter
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
time.Sleep(1000 * time.Millisecond)
sub.Unsubscribe()

// Next: 1 (success, counter resets)
// Next: 2 (success, counter resets)
// Next: 3 (success, counter resets)
// (would retry 3 times after failure since counter reset after each success)
```

### Network request simulation

```go
type Response struct {
    Data string
    Err  error
}

simulateAPICall := func() Observable[Response] {
    return ro.Defer(func() Observable[Response] {
        // Simulate intermittent network failures
        if rand.Intn(5) != 0 { // 80% failure rate
            return ro.Just(Response{Err: errors.New("network timeout")})
        }
        return ro.Just(Response{Data: "api_response"})
    })
}

obs := ro.Pipe[Response, string](
    simulateAPICall(),
    ro.RetryWithConfig[Response](RetryConfig{
        MaxRetries:     10,
        Delay:          200 * time.Millisecond,
        ResetOnSuccess: true,
    }),
    ro.Map(func(r Response) string {
        if r.Err != nil {
            return "error: " + r.Err.Error()
        }
        return "success: " + r.Data
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(3 * time.Second)
sub.Unsubscribe()

// Will keep retrying until successful or max retries reached
// Expected: "success: api_response" (eventually)
```