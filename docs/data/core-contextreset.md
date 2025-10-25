---
name: ContextReset
slug: contextreset
sourceRef: operator_context.go#L129
type: core
category: context
signatures:
  - "func ContextReset[T any](newCtx context.Context)"
playUrl: https://go.dev/play/p/PgvV0SejJpH
variantHelpers:
  - core#context#contextreset
similarHelpers:
  - core#context#contextwithvalue
  - core#context#contextmap
position: 30
---

Replaces the context with a new one (or context.Background() if nil) for each item in the observable sequence.

```go
// First add some context values
obs := ro.Pipe[string, string](
    ro.Just("data1", "data2"),
    ro.ContextWithValue[string]("oldKey", "oldValue"),
    ro.ContextReset[string](context.Background()), // Reset to empty context
    ro.Map(func(s string) string {
        return fmt.Sprintf("Processed: %s", s)
    }),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value string) {
        oldValue := ctx.Value("oldKey")
        fmt.Printf("Next: %s (oldKey: %v)\n", value, oldValue)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: Processed: data1 (oldKey: <nil>)
// Next: Processed: data2 (oldKey: <nil>)
// Completed
```

### With custom new context

```go
newCtx := context.WithValue(context.Background(), "newKey", "newValue")

obs := ro.Pipe[string, string](
    ro.Just("item1", "item2"),
    ro.ContextWithValue[string]("originalKey", "originalValue"),
    ro.ContextReset[string](newCtx),
    ro.Map(func(s string) string {
        return fmt.Sprintf("Item: %s", s)
    }),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value string) {
        original := ctx.Value("originalKey")
        newKey := ctx.Value("newKey")
        fmt.Printf("Next: %s (original: %v, new: %v)\n", value, original, newKey)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: Item: item1 (original: <nil>, new: newValue)
// Next: Item: item2 (original: <nil>, new: newValue)
// Completed
```

### With context timeout reset

```go
// Original context with timeout
timeoutCtx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
defer cancel()

// New context with longer timeout
newTimeoutCtx, newCancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
defer newCancel()

obs := ro.Pipe[string, string](
    ro.Just("slow_operation"),
    ro.ContextReset[string](newTimeoutCtx),
    ro.ThrowOnContextCancel[string](),
    ro.Map(func(s string) string {
        time.Sleep(100 * time.Millisecond) // Would timeout with original context
        return fmt.Sprintf("Completed %s", s)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: "Completed: slow_operation"
// Completed (succeeds because context was reset to longer timeout)
```

### With nil context (resets to Background)

```go
obs := ro.Pipe[string, string](
    ro.Just("reset_test"),
    ro.ContextWithValue[string]("someKey", "someValue"),
    ro.ContextReset[string](nil), // Resets to context.Background()
    ro.Map(func(s string) string {
        return fmt.Sprintf("Reset: %s", s)
    }),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value string) {
        someKey := ctx.Value("someKey")
        fmt.Printf("Next: %s (someKey: %v)\n", value, someKey)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: Reset: reset_test (someKey: <nil>)
// Completed
```

### With context isolation in async operations

```go
processAsync := func(item string) ro.Observable[string] {
    return ro.Defer(func() ro.Observable[string] {
        time.Sleep(50 * time.Millisecond)
        return ro.Just(fmt.Sprintf("Async: %s", item))
    })
}

obs := ro.Pipe[string, string](
    ro.Just("item1", "item2"),
    ro.ContextWithValue[string]("sharedID", "shared-123"),
    // Each item gets fresh context for async processing
    ro.ContextReset[string](context.Background()),
    ro.MergeMap(processAsync),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value string) {
        sharedID := ctx.Value("sharedID")
        fmt.Printf("Next: %s (sharedID: %v)\n", value, sharedID)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
time.Sleep(200 * time.Millisecond)
sub.Unsubscribe()

// Next: Async: item1 (sharedID: <nil>)
// Next: Async: item2 (sharedID: <nil>)
// Completed
```

### With multiple context transformations

```go
initialCtx := context.WithValue(context.Background(), "step1", "value1")
step2Ctx := context.WithValue(context.Background(), "step2", "value2")
step3Ctx := context.WithValue(context.Background(), "step3", "value3")

obs := ro.Pipe[string, string](
    ro.Just("multi_context"),
    ro.ContextWithValue[string]("extra", "extra_value"),
    ro.ContextReset[string](initialCtx),
    ro.ContextWithValue[string]("added", "added_value"),
    ro.ContextReset[string](step2Ctx),
    ro.Map(func(s string) string {
        return fmt.Sprintf("Final: %s", s)
    }),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value string) {
        step1 := ctx.Value("step1")
        step2 := ctx.Value("step2")
        step3 := ctx.Value("step3")
        extra := ctx.Value("extra")
        added := ctx.Value("added")
        fmt.Printf("Next: %s (step1: %v, step2: %v, step3: %v, extra: %v, added: %v)\n",
            value, step1, step2, step3, extra, added)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: Final: multi_context (step1: <nil>, step2: value2, step3: <nil>, extra: <nil>, added: <nil>)
// Completed
```

### With context reset for error isolation

```go
obs := ro.Pipe[string, string](
    ro.Just("operation1", "operation2"),
    ro.ContextWithValue[string]("requestID", "req-abc"),
    ro.MapErr(func(s string) (string, error) {
        if s == "operation2" {
            return "", fmt.Errorf("error in %s", s)
        }
        return s, nil
    }),
    ro.Catch(func(err error) Observable[string] {
        // Reset context for fallback to avoid leaking sensitive data
        return ro.Pipe[string, string](
            ro.Just(fmt.Sprintf("Fallback: %v", err)),
            ro.ContextReset[string](context.Background()),
        )
    }),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value string) {
        requestID := ctx.Value("requestID")
        fmt.Printf("Next: %s (requestID: %v)\n", value, requestID)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: operation1 (requestID: req-abc)
// Next: Fallback: error in operation2 (requestID: <nil>)
// Completed
```