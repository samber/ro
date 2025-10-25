---
name: Do
slug: do
sourceRef: operator_utility.go#L74
type: core
category: utility
signatures:
  - "func Do[T any](onNext func(value T), onError func(err error), onComplete func())"
  - "func DoWithContext[T any](onNext func(ctx context.Context, value T), onError func(ctx context.Context, err error), onComplete func(ctx context.Context))"
  - "func DoOnNext[T any](onNext func(value T))"
  - "func DoOnNextWithContext[T any](onNext func(ctx context.Context, value T))"
  - "func DoOnError[T any](onError func(err error))"
  - "func DoOnErrorWithContext[T any](onError func(ctx context.Context, err error))"
  - "func DoOnComplete[T any](onComplete func())"
  - "func DoOnCompleteWithContext[T any](onComplete func(ctx context.Context))"
  - "func DoOnSubscribe[T any](onSubscribe func())"
  - "func DoOnSubscribeWithContext[T any](onSubscribe func(ctx context.Context))"
  - "func DoOnFinalize[T any](onFinalize func())"
playUrl: https://go.dev/play/p/s_BSHgxdjUR
variantHelpers:
  - core#utility#do
  - core#utility#dowithcontext
  - core#utility#doonnext
  - core#utility#doonnextwithcontext
  - core#utility#doonerror
  - core#utility#doonerrorwithcontext
  - core#utility#dooncomplete
  - core#utility#dooncompletewithcontext
  - core#utility#doonsubscribe
  - core#utility#doonsubscribewithcontext
  - core#utility#doonfinalize
similarHelpers:
  - core#utility#tap
position: 0
---

Performs side effects for each emission from an Observable, without modifying the emitted values.

```go
var nextCount, errorCount, completeCount int

obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.Do(
        func(value int) {
            nextCount++
            fmt.Printf("Next: %d\n", value)
        },
        func(err error) {
            errorCount++
            fmt.Printf("Error: %v\n", err)
        },
        func() {
            completeCount++
            fmt.Println("Complete")
        },
    ),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Completed

fmt.Printf("Counts: next=%d, error=%d, complete=%d\n", nextCount, errorCount, completeCount)
// Counts: next=3, error=0, complete=1
```

### With context

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.DoWithContext(
        func(ctx context.Context, value int) {
            fmt.Printf("Next with context: %d\n", value)
        },
        func(ctx context.Context, err error) {
            fmt.Printf("Error with context: %v\n", err)
        },
        func(ctx context.Context) {
            fmt.Println("Complete with context")
        },
    ),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()
```

### DoOnNext

```go
var nextValues []int

obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.DoOnNext(func(value int) {
        nextValues = append(nextValues, value)
        fmt.Printf("Received: %d\n", value)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Received: 1
// Received: 2
// Received: 3

fmt.Printf("Collected values: %v\n", nextValues)
// Collected values: [1 2 3]
```

### DoOnError

```go
var lastError error

obs := ro.Pipe[int, int](
    ro.Throw[int](fmt.Errorf("test error")),
    ro.DoOnError(func(err error) {
        lastError = err
        fmt.Printf("Error captured: %v\n", err)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Error captured: test error
// Error: test error
```

### DoOnComplete

```go
var completed bool

obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.DoOnComplete(func() {
        completed = true
        fmt.Println("Stream completed")
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Stream completed
```

### DoOnSubscribe

```go
var subscribed bool

obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.DoOnSubscribe(func() {
        subscribed = true
        fmt.Println("Subscribed to observable")
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Subscribed to observable
```

### DoOnFinalize

```go
var finalized bool

obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.DoOnFinalize(func() {
        finalized = true
        fmt.Println("Observable finalized")
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Observable finalized
```