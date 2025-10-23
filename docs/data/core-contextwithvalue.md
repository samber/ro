---
name: ContextWithValue
slug: contextwithvalue
sourceRef: operator_context.go#L24
type: core
category: context
signatures:
  - "func ContextWithValue[T any](k any, v any)"
playUrl:
variantHelpers:
  - core#context#contextwithvalue
similarHelpers:
  - core#context#contextmap
  - core#context#contextreset
position: 0
---

Adds a key-value pair to the context of each item in the observable sequence.

```go
obs := ro.Pipe[string, string](
    ro.Just("request1", "request2"),
    ro.ContextWithValue[string]("requestID", "req-123"),
    ro.Map(func(s string) string {
        return fmt.Sprintf("Processing %s", s)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Each item now has context with requestID: "req-123"
// Next: "Processing request1"
// Next: "Processing request2"
// Completed
```

### With context extraction in downstream operators

```go
obs := ro.Pipe[string, string](
    ro.Just("data1", "data2"),
    ro.ContextWithValue[string]("userID", 42),
    ro.Map(func(s string) string {
        return fmt.Sprintf("User data: %s", s)
    }),
)

// Extract context in subscription
sub := obs.Subscribe(ro.NewObserver(
    func(value string) {
        userID := ctx.Value("userID")
        fmt.Printf("Next: %s (userID: %v)\n", value, userID)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: User data: data1 (userID: 42)
// Next: User data: data2 (userID: 42)
// Completed
```

### With multiple context values

```go
obs := ro.Pipe[string, string](
    ro.Just("task1", "task2"),
    ro.ContextWithValue[string]("traceID", "trace-abc"),
    ro.ContextWithValue[string]("sessionID", "session-xyz"),
    ro.Map(func(s string) string {
        return fmt.Sprintf("Executing %s", s)
    }),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value string) {
        traceID := ctx.Value("traceID")
        sessionID := ctx.Value("sessionID")
        fmt.Printf("Next: %s (trace: %v, session: %v)\n", value, traceID, sessionID)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: Executing task1 (trace: trace-abc, session: session-xyz)
// Next: Executing task2 (trace: trace-abc, session: session-xyz)
// Completed
```

### With structured context values

```go
type RequestMetadata struct {
    TraceID   string
    Timestamp time.Time
    UserID    int
}

metadata := RequestMetadata{
    TraceID:   "req-456",
    Timestamp: time.Now(),
    UserID:    789,
}

obs := ro.Pipe[string, string](
    ro.Just("api_call1", "api_call2"),
    ro.ContextWithValue[string]("metadata", metadata),
    ro.Map(func(s string) string {
        return fmt.Sprintf("API call: %s", s)
    }),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value string) {
        if meta, ok := ctx.Value("metadata").(RequestMetadata); ok {
            fmt.Printf("Next: %s (user: %d, trace: %s)\n",
                value, meta.UserID, meta.TraceID)
        }
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: API call: api_call1 (user: 789, trace: req-456)
// Next: API call: api_call2 (user: 789, trace: req-456)
// Completed
```

### With context-aware error handling

```go
obs := ro.Pipe[string, string](
    ro.Just("critical_op1", "critical_op2"),
    ro.ContextWithValue[string]("operationType", "high_priority"),
    ro.MapErr(func(s string) (string, error) {
        if s == "critical_op2" {
            return "", fmt.Errorf("failed operation: %s", s)
        }
        return s, nil
    }),
    ro.Catch(func(err error) Observable[string] {
        return ro.Just(fmt.Sprintf("Fallback for error: %v", err))
    }),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value string) {
        opType := ctx.Value("operationType")
        fmt.Printf("Next: %s (type: %v)\n", value, opType)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: critical_op1 (type: high_priority)
// Next: Fallback for error: failed operation: critical_op2 (type: high_priority)
// Completed
```

### With nested context in async operations

```go
processAsync := func(ctx context.Context, item string) ro.Observable[string] {
    traceID := ctx.Value("traceID")
    return ro.Defer(func() ro.Observable[string] {
        time.Sleep(50 * time.Millisecond) // Simulate async work
        return ro.Just(fmt.Sprintf("Processed %s (trace: %v)", item, traceID))
    })
}

obs := ro.Pipe[string, string](
    ro.Just("item1", "item2"),
    ro.ContextWithValue[string]("traceID", "async-trace-789"),
    ro.MergeMap(func(item string) ro.Observable[string] {
        return ro.Defer(func() ro.Observable[string] {
            // Extract current context for async operation
            return processAsync(context.Background(), item)
        })
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(200 * time.Millisecond)
sub.Unsubscribe()

// Next: Processed item1 (trace: async-trace-789)
// Next: Processed item2 (trace: async-trace-789)
// Completed
```