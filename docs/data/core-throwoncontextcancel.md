---
name: ThrowOnContextCancel
slug: throwoncontextcancel
sourceRef: operator_context.go#L196
type: core
category: context
signatures:
  - "func ThrowOnContextCancel[T any]()"
playUrl:
variantHelpers:
  - core#context#throwoncontextcancel
similarHelpers:
  - core#context#contextwithtimeout
  - core#context#contextwithdeadline
position: 50
---

Throws an error if the context is canceled. Should be chained after timeout/deadline operators to handle context cancellation.

```go
obs := ro.Pipe[string, string](
    ro.Just("timed_operation"),
    ro.ContextWithTimeout[string](100 * time.Millisecond),
    ro.ThrowOnContextCancel[string](),
    ro.Map(func(s string) string {
        time.Sleep(150 * time.Millisecond) // Will exceed timeout
        return fmt.Sprintf("Completed %s", s)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Error: context deadline exceeded
```

### With successful completion

```go
obs := ro.Pipe[string, string](
    ro.Just("fast_operation"),
    ro.ContextWithTimeout[string](200 * time.Millisecond),
    ro.ThrowOnContextCancel[string](),
    ro.Map(func(s string) string {
        time.Sleep(100 * time.Millisecond) // Completes within timeout
        return fmt.Sprintf("Success: %s", s)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: "Success: fast_operation"
// Completed
```

### With manual context cancellation

```go
ctx, cancel := context.WithCancel(context.Background())

obs := ro.Pipe[string, string](
    ro.Just("cancelable_operation"),
    ro.ContextReset[string](ctx),
    ro.ThrowOnContextCancel[string](),
    ro.Map(func(s string) string {
        time.Sleep(200 * time.Millisecond)
        return fmt.Sprintf("Result: %s", s)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())

// Cancel after short delay
go func() {
    time.Sleep(100 * time.Millisecond)
    cancel()
}()

time.Sleep(300 * time.Millisecond)
sub.Unsubscribe()

// Error: context canceled
```

### With deadline

```go
deadline := time.Now().Add(100 * time.Millisecond)
obs := ro.Pipe[string, string](
    ro.Just("deadline_operation"),
    ro.ContextWithDeadline[string](deadline),
    ro.ThrowOnContextCancel[string](),
    ro.Map(func(s string) string {
        time.Sleep(150 * time.Millisecond) // Will exceed deadline
        return fmt.Sprintf("Deadline result: %s", s)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Error: context deadline exceeded
```

### With retry on cancellation

```go
obs := ro.Pipe[string, string](
    ro.Defer(func() Observable[string] {
        ctx, cancel := context.WithCancel(context.Background())

        // Auto-cancel after short delay
        go func() {
            time.Sleep(50 * time.Millisecond)
            cancel()
        }()

        return ro.Pipe[string, string](
            ro.Just("retryable_operation"),
            ro.ContextReset[string](ctx),
            ro.ThrowOnContextCancel[string](),
            ro.Map(func(s string) string {
                time.Sleep(100 * time.Millisecond) // Will be cancelled
                return fmt.Sprintf("Success: %s", s)
            }),
        )
    }),
    ro.RetryWithConfig[string](RetryConfig{
        MaxRetries: 3,
        Delay:      100 * time.Millisecond,
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(500 * time.Millisecond)
sub.Unsubscribe()

// Will retry up to 3 times when context is cancelled
// If all attempts are cancelled, final error will be context canceled
```

### With graceful error handling

```go
obs := ro.Pipe[string, string](
    ro.Just("graceful_operation"),
    ro.ContextWithTimeout[string](100 * time.Millisecond),
    ro.ThrowOnContextCancel[string](),
    ro.Catch(func(err error) Observable[string] {
        if errors.Is(err, context.Canceled) {
            return ro.Just("Operation was cancelled gracefully")
        }
        if errors.Is(err, context.DeadlineExceeded) {
            return ro.Just("Operation timed out - using cached result")
        }
        return ro.Throw[string](err)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: "Operation timed out - using cached result"
// Completed
```

### With async operations

```go
processAsync := func(item string) Observable[string] {
    return ro.Defer(func() Observable[string] {
        time.Sleep(150 * time.Millisecond) // Simulate slow async work
        return ro.Just(fmt.Sprintf("Async result: %s", item))
    })
}

obs := ro.Pipe[string, string](
    ro.Just("item1", "item2"),
    ro.ContextWithTimeout[string](100 * time.Millisecond),
    ro.ThrowOnContextCancel[string](),
    ro.MergeMap(processAsync),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(300 * time.Millisecond)
sub.Unsubscribe()

// Error: context deadline exceeded (async operations take longer than timeout)
```

### With complex pipeline and multiple cancellation points

```go
obs := ro.Pipe[string, string](
    ro.Just("complex_pipeline"),
    ro.ContextWithValue[string]("requestID", "req-123"),
    ro.ContextWithTimeout[string](80 * time.Millisecond),
    ro.ThrowOnContextCancel[string](),
    ro.Map(func(s string) string {
        time.Sleep(50 * time.Millisecond)
        return fmt.Sprintf("Step1: %s", s)
    }),
    ro.Map(func(s string) string {
        time.Sleep(50 * time.Millisecond) // Will exceed timeout here
        return fmt.Sprintf("Step2: %s", s)
    }),
    ro.Map(func(s string) string {
        return fmt.Sprintf("Final: %s", s)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Error: context deadline exceeded (pipeline cancelled during second map operation)
```

### With context cancellation from parent

```go
parentCtx, parentCancel := context.WithCancel(context.Background())

obs := ro.Pipe[string, string](
    ro.Just("child_operation"),
    ro.ContextReset[string](parentCtx),
    ro.ThrowOnContextCancel[string](),
    ro.Map(func(s string) string {
        time.Sleep(200 * time.Millisecond)
        return fmt.Sprintf("Child result: %s", s)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())

// Cancel from parent context
go func() {
    time.Sleep(100 * time.Millisecond)
    parentCancel()
}()

time.Sleep(300 * time.Millisecond)
sub.Unsubscribe()

// Error: context canceled (inherited from parent context)
```