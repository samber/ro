---
name: ContextWithDeadline
slug: contextwithdeadline
sourceRef: operator_context.go#L91
type: core
category: context
signatures:
  - "func ContextWithDeadline[T any](deadline time.Time)"
  - "func ContextWithDeadlineCause[T any](deadline time.Time, cause error)"
playUrl:
variantHelpers:
  - core#context#contextwithdeadline
  - core#context#contextwithdeadlinecause
similarHelpers:
  - core#context#contextwithtimeout
  - core#context#throwoncontextcancel
position: 20
---

Adds a deadline to the context of each item in the observable sequence. Should be chained with ThrowOnContextCancel to handle deadline errors.

```go
deadline := time.Now().Add(100 * time.Millisecond)
obs := ro.Pipe[string, string](
    ro.Just("deadline_operation"),
    ro.ContextWithDeadline[string](deadline),
    ro.ThrowOnContextCancel[string](),
    ro.Map(func(s string) string {
        time.Sleep(150 * time.Millisecond) // Simulate slow operation
        return fmt.Sprintf("Completed %s", s)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Error: context deadline exceeded (operation exceeded deadline)
```

### With successful completion before deadline

```go
deadline := time.Now().Add(200 * time.Millisecond)
obs := ro.Pipe[string, string](
    ro.Just("fast_operation"),
    ro.ContextWithDeadline[string](deadline),
    ro.ThrowOnContextCancel[string](),
    ro.Map(func(s string) string {
        time.Sleep(100 * time.Millisecond) // Completes before deadline
        return fmt.Sprintf("Success: %s", s)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: "Success: fast_operation"
// Completed
```

### With ContextWithDeadlineCause

```go
deadline := time.Now().Add(50 * time.Millisecond)
deadlineError := errors.New("processing deadline exceeded")
obs := ro.Pipe[string, string](
    ro.Just("data_processing"),
    ro.ContextWithDeadlineCause[string](deadline, deadlineError),
    ro.ThrowOnContextCancel[string](),
    ro.Map(func(s string) string {
        time.Sleep(100 * time.Millisecond) // Will exceed deadline
        return fmt.Sprintf("Result: %s", s)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Error: processing deadline exceeded (custom cause error)
```

### With future deadline

```go
// Set deadline for 5 seconds from now
deadline := time.Now().Add(5 * time.Second)
obs := ro.Pipe[string, string](
    ro.Just("long_operation"),
    ro.ContextWithDeadline[string](deadline),
    ro.ThrowOnContextCancel[string](),
    ro.Map(func(s string) string {
        time.Sleep(2 * time.Second) // Completes well before deadline
        return fmt.Sprintf("Long operation completed: %s", s)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(2500 * time.Millisecond)
sub.Unsubscribe()

// Next: "Long operation completed: long_operation"
// Completed
```

### With multiple operations and shared deadline

```go
deadline := time.Now().Add(200 * time.Millisecond)
obs := ro.Pipe[string, string](
    ro.Just("op1", "op2", "op3"),
    ro.ContextWithDeadline[string](deadline),
    ro.ThrowOnContextCancel[string](),
    ro.Map(func(s string) string {
        if s == "op2" {
            time.Sleep(250 * time.Millisecond) // This will exceed deadline
        }
        return fmt.Sprintf("Processed: %s", s)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: "Processed: op1"
// Error: context deadline exceeded (on op2)
```

### With deadline in the past

```go
// Set deadline to past time (immediate timeout)
deadline := time.Now().Add(-1 * time.Hour)
obs := ro.Pipe[string, string](
    ro.Just("immediate_timeout"),
    ro.ContextWithDeadline[string](deadline),
    ro.ThrowOnContextCancel[string](),
    ro.Map(func(s string) string {
        return fmt.Sprintf("This won't execute: %s", s)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Error: context deadline exceeded (deadline already passed)
```

### With deadline-based batch processing

```go
type BatchJob struct {
    ID   string
    Work func() string
}

jobs := []BatchJob{
    {"job1", func() string { time.Sleep(50 * time.Millisecond); return "job1 result" }},
    {"job2", func() string { time.Sleep(150 * time.Millisecond); return "job2 result" }},
    {"job3", func() string { time.Sleep(200 * time.Millisecond); return "job3 result" }},
}

deadline := time.Now().Add(180 * time.Millisecond)
obs := ro.Pipe[BatchJob, string](
    ro.FromSlice(jobs),
    ro.ContextWithDeadline[BatchJob](deadline),
    ro.ThrowOnContextCancel[BatchJob](),
    ro.Map(func(job BatchJob) string {
        return job.Work() // Execute the job
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(400 * time.Millisecond)
sub.Unsubscribe()

// Next: "job1 result"
// Next: "job2 result"
// Error: context deadline exceeded (job3 times out)
```

### With deadline retry mechanism

```go
attempt := 0
obs := ro.Pipe[string, string](
    ro.Defer(func() Observable[string] {
        attempt++
        deadline := time.Now().Add(100 * time.Millisecond)
        return ro.Pipe[string, string](
            ro.Just(fmt.Sprintf("attempt_%d", attempt)),
            ro.ContextWithDeadline[string](deadline),
            ro.ThrowOnContextCancel[string](),
            ro.Map(func(s string) string {
                time.Sleep(120 * time.Millisecond) // Always exceed deadline
                return fmt.Sprintf("Success: %s", s)
            }),
        )
    }),
    ro.RetryWithConfig[string](ro.RetryConfig{
        MaxRetries: 2,
        Delay:      50 * time.Millisecond,
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(500 * time.Millisecond)
sub.Unsubscribe()

// Will retry twice (attempt_1 and attempt_2 both timeout)
// Error: context deadline exceeded
```

### With graceful deadline handling

```go
deadline := time.Now().Add(100 * time.Millisecond)
obs := ro.Pipe[string, string](
    ro.Just("graceful_operation"),
    ro.ContextWithDeadline[string](deadline),
    ro.ThrowOnContextCancel[string](),
    ro.Catch(func(err error) Observable[string] {
        if errors.Is(err, context.DeadlineExceeded) {
            return ro.Just("Operation cancelled due to deadline - data saved")
        }
        return ro.Throw[string](err)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: "Operation cancelled due to deadline - data saved"
// Completed
```