---
name: ContextMap
slug: contextmap
sourceRef: operator_context.go#L160
type: core
category: context
signatures:
  - "func ContextMap[T any](project func(ctx context.Context) context.Context)"
  - "func ContextMapI[T any](project func(ctx context.Context, index int64) context.Context)"
playUrl:
variantHelpers:
  - core#context#contextmap
  - core#context#contextmapi
similarHelpers:
  - core#context#contextwithvalue
  - core#context#contextreset
position: 40
---

Transforms the context using a project function for each item in the observable sequence.

```go
obs := ro.Pipe[string, string](
    ro.Just("item1", "item2"),
    ro.ContextMap[string](func(ctx context.Context) context.Context {
        return context.WithValue(ctx, "processed", true)
    }),
    ro.Map(func(s string) string {
        return fmt.Sprintf("Processed: %s", s)
    }),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value string) {
        processed := ctx.Value("processed")
        fmt.Printf("Next: %s (processed: %v)\n", value, processed)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: Processed: item1 (processed: true)
// Next: Processed: item2 (processed: true)
// Completed
```

### ContextMapI with index

```go
obs := ro.Pipe[string, string](
    ro.Just("a", "b", "c"),
    ro.ContextMapI[string](func(ctx context.Context, index int64) context.Context {
        return context.WithValue(ctx, "itemIndex", index)
    }),
    ro.Map(func(s string) string {
        return fmt.Sprintf("Item: %s", s)
    }),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value string) {
        itemIndex := ctx.Value("itemIndex")
        fmt.Printf("Next: %s (index: %v)\n", value, itemIndex)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: Item: a (index: 0)
// Next: Item: b (index: 1)
// Next: Item: c (index: 2)
// Completed
```

### With context transformation chain

```go
obs := ro.Pipe[string, string](
    ro.Just("data"),
    ro.ContextWithValue[string]("userID", 123),
    ro.ContextMap[string](func(ctx context.Context) context.Context {
        // Add timestamp to context
        return context.WithValue(ctx, "timestamp", time.Now())
    }),
    ro.ContextMap[string](func(ctx context.Context) context.Context {
        // Add request ID based on existing context
        userID := ctx.Value("userID")
        requestID := fmt.Sprintf("req-%v-%d", userID, time.Now().Unix())
        return context.WithValue(ctx, "requestID", requestID)
    }),
    ro.Map(func(s string) string {
        return fmt.Sprintf("Data: %s", s)
    }),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value string) {
        userID := ctx.Value("userID")
        timestamp := ctx.Value("timestamp")
        requestID := ctx.Value("requestID")
        fmt.Printf("Next: %s (userID: %v, timestamp: %v, requestID: %v)\n",
            value, userID, timestamp, requestID)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: Data: data (userID: 123, timestamp: <current_time>, requestID: req-123-<timestamp>)
// Completed
```

### With context-based routing

```go
obs := ro.Pipe[string, string](
    ro.Just("request1", "request2", "request3"),
    ro.ContextMapI[string](func(ctx context.Context, index int64) context.Context {
        // Route even and odd items to different contexts
        if index%2 == 0 {
            return context.WithValue(ctx, "route", "primary")
        }
        return context.WithValue(ctx, "route", "secondary")
    }),
    ro.Map(func(s string) string {
        return fmt.Sprintf("Processed: %s", s)
    }),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value string) {
        route := ctx.Value("route")
        fmt.Printf("Next: %s (route: %v)\n", value, route)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: Processed: request1 (route: primary)
// Next: Processed: request2 (route: secondary)
// Next: Processed: request3 (route: primary)
// Completed
```

### With context modification based on item content

```go
obs := ro.Pipe[string, string](
    ro.Just("urgent", "normal", "critical", "low"),
    ro.ContextMap[string](func(ctx context.Context) context.Context {
        // This would typically need access to the item value
        // For this example, we'll simulate context modification
        return context.WithValue(ctx, "processedAt", time.Now())
    }),
    ro.Map(func(s string) string {
        priority := "normal"
        if s == "urgent" || s == "critical" {
            priority = "high"
        }
        return fmt.Sprintf("%s (%s priority)", s, priority)
    }),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value string) {
        processedAt := ctx.Value("processedAt")
        fmt.Printf("Next: %s (processedAt: %v)\n", value, processedAt)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: urgent (high priority) (processedAt: <timestamp>)
// Next: normal (normal priority) (processedAt: <timestamp>)
// Next: critical (high priority) (processedAt: <timestamp>)
// Next: low (normal priority) (processedAt: <timestamp>)
// Completed
```

### With context inheritance and modification

```go
baseCtx := context.WithValue(context.Background(), "sessionID", "session-abc")
baseCtx = context.WithValue(baseCtx, "userID", 456)

obs := ro.Pipe[string, string](
    ro.Just("operation1", "operation2"),
    ro.ContextReset[string](baseCtx),
    ro.ContextMap[string](func(ctx context.Context) context.Context {
        // Inherit from base context and add operation-specific data
        return context.WithValue(ctx, "operationID", fmt.Sprintf("op-%d", time.Now().UnixNano()))
    }),
    ro.ContextMapI[string](func(ctx context.Context, index int64) context.Context {
        // Add sequential information
        return context.WithValue(ctx, "sequence", index+1)
    }),
    ro.Map(func(s string) string {
        return fmt.Sprintf("Executed: %s", s)
    }),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value string) {
        sessionID := ctx.Value("sessionID")
        userID := ctx.Value("userID")
        operationID := ctx.Value("operationID")
        sequence := ctx.Value("sequence")
        fmt.Printf("Next: %s (session: %v, user: %v, opID: %v, seq: %v)\n",
            value, sessionID, userID, operationID, sequence)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: Executed: operation1 (session: session-abc, user: 456, opID: op-<nanotime>, seq: 1)
// Next: Executed: operation2 (session: session-abc, user: 456, opID: op-<nanotime>, seq: 2)
// Completed
```

### With conditional context transformation

```go
obs := ro.Pipe[string, string](
    ro.Just("debug", "info", "error", "warning"),
    ro.ContextMapI[string](func(ctx context.Context, index int64) context.Context {
        // Transform context based on index
        ctx = context.WithValue(ctx, "index", index)
        if index >= 2 { // error and warning
            return context.WithValue(ctx, "severity", "high")
        }
        return context.WithValue(ctx, "severity", "low")
    }),
    ro.Map(func(s string) string {
        return fmt.Sprintf("Log: %s", s)
    }),
)

sub := obs.Subscribe(ro.NewObserver(
    func(ctx context.Context, value string) {
        index := ctx.Value("index")
        severity := ctx.Value("severity")
        fmt.Printf("Next: %s (index: %v, severity: %v)\n", value, index, severity)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: Log: debug (index: 0, severity: low)
// Next: Log: info (index: 1, severity: low)
// Next: Log: error (index: 2, severity: high)
// Next: Log: warning (index: 3, severity: high)
// Completed
```