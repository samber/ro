---
name: ContextWithTimeout
slug: contextwithtimeout
sourceRef: operator_context.go#L53
type: core
category: context
signatures:
  - "func ContextWithTimeout[T any](timeout time.Duration)"
  - "func ContextWithTimeoutCause[T any](timeout time.Duration, cause error)"
playUrl: https://go.dev/play/p/1qijKGsyn0D
variantHelpers:
  - core#context#contextwithtimeout
  - core#context#contextwithtimeoutcause
similarHelpers:
  - core#context#contextwithdeadline
  - core#context#throwoncontextcancel
position: 10
---

Adds a timeout to the context of each item in the observable sequence. Should be chained with ThrowOnContextCancel to handle timeout errors.

```go
obs := ro.Pipe[string, string](
    ro.Just("slow_operation"),
    ro.ContextWithTimeout[string](100 * time.Millisecond),
    ro.ThrowOnContextCancel[string](),
    ro.Map(func(s string) string {
        time.Sleep(150 * time.Millisecond) // Simulate slow operation
        return fmt.Sprintf("Completed %s", s)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Error: context deadline exceeded (operation took longer than 100ms timeout)
```

### With successful completion within timeout

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

### With ContextWithTimeoutCause

```go
timeoutError := errors.New("operation timed out")
obs := ro.Pipe[string, string](
    ro.Just("data_processing"),
    ro.ContextWithTimeoutCause[string](50 * time.Millisecond, timeoutError),
    ro.ThrowOnContextCancel[string](),
    ro.Map(func(s string) string {
        time.Sleep(100 * time.Millisecond) // Will timeout
        return fmt.Sprintf("Result: %s", s)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Error: operation timed out (custom cause error)
```

### With multiple operations and timeout

```go
obs := ro.Pipe[string, string](
    ro.Just("op1", "op2", "op3"),
    ro.ContextWithTimeout[string](150 * time.Millisecond),
    ro.ThrowOnContextCancel[string](),
    ro.Map(func(s string) string {
        if s == "op2" {
            time.Sleep(200 * time.Millisecond) // This one will timeout
        }
        return fmt.Sprintf("Processed: %s", s)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: "Processed: op1"
// Error: context deadline exceeded (on op2)
```

### With retry mechanism after timeout

```go
obs := ro.Pipe[string, string](
    ro.Defer(func() Observable[string] {
        return ro.Pipe[string, string](
            ro.Just("retry_operation"),
            ro.ContextWithTimeout[string](100 * time.Millisecond),
            ro.ThrowOnContextCancel[string](),
            ro.Map(func(s string) string {
                time.Sleep(150 * time.Millisecond) // Always timeout
                return fmt.Sprintf("Success: %s", s)
            }),
        )
    }),
    ro.RetryWithConfig[string](ro.RetryConfig{
        MaxRetries: 3,
        Delay:      50 * time.Millisecond,
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(500 * time.Millisecond)
sub.Unsubscribe()

// Will retry up to 3 times with 50ms delays
// Error: context deadline exceeded (if all retries timeout)
```

### With async operations

```go
processItem := func(item string) Observable[string] {
    return ro.Defer(func() Observable[string] {
        time.Sleep(80 * time.Millisecond) // Simulate async processing
        return ro.Just(fmt.Sprintf("Async result: %s", item))
    })
}

obs := ro.Pipe[string, string](
    ro.Just("item1", "item2"),
    ro.ContextWithTimeout[string](60 * time.Millisecond),
    ro.ThrowOnContextCancel[string](),
    ro.MergeMap(processItem),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(200 * time.Millisecond)
sub.Unsubscribe()

// Error: context deadline exceeded (async processing takes longer than timeout)
```

### With different timeout values

```go
type Task struct {
    Name     string
    Duration time.Duration
}

tasks := []Task{
    {"quick", 50 * time.Millisecond},
    {"medium", 100 * time.Millisecond},
    {"slow", 200 * time.Millisecond},
}

obs := ro.Pipe[Task, string](
    ro.FromSlice(tasks),
    ro.ContextWithTimeout[Task](150 * time.Millisecond),
    ro.ThrowOnContextCancel[Task](),
    ro.Map(func(task Task) string {
        time.Sleep(task.Duration)
        return fmt.Sprintf("Task %s completed", task.Name)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(500 * time.Millisecond)
sub.Unsubscribe()

// Next: "Task quick completed"
// Next: "Task medium completed"
// Error: context deadline exceeded (slow task times out)
```

### With context timeout handling

```go
obs := ro.Pipe[string, string](
    ro.Just("timed_operation"),
    ro.ContextWithTimeout[string](100 * time.Millisecond),
    ro.ThrowOnContextCancel[string](),
    ro.Catch(func(err error) Observable[string] {
        if errors.Is(err, context.DeadlineExceeded) {
            return ro.Just("Operation timed out - using fallback")
        }
        return ro.Throw[string](err)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: "Operation timed out - using fallback"
// Completed
```