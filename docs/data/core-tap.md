---
name: Tap
slug: tap
sourceRef: operator_utility.go#L30
type: core
category: utility
signatures:
  - "func Tap[T any](onNext func(value T), onError func(err error), onComplete func())"
  - "func TapWithContext[T any](onNext func(ctx context.Context, value T), onError func(ctx context.Context, err error), onComplete func(ctx context.Context))"
  - "func TapOnNext[T any](onNext func(value T))"
  - "func TapOnNextWithContext[T any](onNext func(ctx context.Context, value T))"
  - "func TapOnError[T any](onError func(err error))"
  - "func TapOnErrorWithContext[T any](onError func(ctx context.Context, err error))"
  - "func TapOnComplete[T any](onComplete func())"
  - "func TapOnCompleteWithContext[T any](onComplete func(ctx context.Context))"
playUrl:
variantHelpers:
  - core#utility#tap
  - core#utility#tapwithcontext
  - core#utility#taponnext
  - core#utility#taponnextwithcontext
  - core#utility#taponerror
  - core#utility#taponerrorwithcontext
  - core#utility#taponcomplete
  - core#utility#taponcompletewithcontext
similarHelpers:
  - core#utility#do
position: 10
---

Allows you to perform side effects for notifications from the source Observable without modifying the emitted items. It mirrors the source Observable and forwards its emissions to the provided observer.

```go
var nextCount, errorCount, completeCount int

obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.Tap(
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
    ro.TapWithContext(
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

### TapOnNext

```go
var nextValues []int

obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.TapOnNext(func(value int) {
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

### TapOnError

```go
var lastError error

obs := ro.Pipe[int, int](
    ro.Throw[int](fmt.Errorf("test error")),
    ro.TapOnError(func(err error) {
        lastError = err
        fmt.Printf("Error captured: %v\n", err)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Error captured: test error
// Error: test error
```

### TapOnComplete

```go
var completed bool

obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.TapOnComplete(func() {
        completed = true
        fmt.Println("Stream completed")
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Stream completed
```

### With context error handling

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.TapOnErrorWithContext(func(ctx context.Context, err error) {
        fmt.Printf("Error with context: %v\n", err)
    }),
    ro.TapOnCompleteWithContext(func(ctx context.Context) {
        fmt.Println("Completed with context")
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()
```

### For debugging

```go
obs := ro.Pipe[int64, int64](
    ro.Interval(100*time.Millisecond),
    ro.Take[int64](3),
    ro.TapOnNext(func(value int64) {
        fmt.Printf("[DEBUG] Received: %d\n", value)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
time.Sleep(500 * time.Millisecond)
sub.Unsubscribe()

// [DEBUG] Received: 0
// [DEBUG] Received: 1
// [DEBUG] Received: 2
```

### With error scenarios

```go
obs := ro.Pipe[int, int](
    ro.Pipe[int, int](
        ro.Just(1, 2, 3),
        ro.MapErr(func(i int) (int, error) {
            if i == 3 {
                return 0, fmt.Errorf("error on 3")
            }
            return i, nil
        }),
    ),
    ro.Tap(
        func(value int) {
            fmt.Printf("Next tap: %d\n", value)
        },
        func(err error) {
            fmt.Printf("Error tap: %v\n", err)
        },
        func() {
            fmt.Println("Complete tap")
        },
    ),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next tap: 1
// Next tap: 2
// Error tap: error on 3
```

### With hot observables

```go
source := ro.Pipe[int64, int64](
    ro.Interval(100*time.Millisecond),
    ro.TapOnNext(func(value int64) {
        fmt.Printf("Source value: %d\n", value)
    }),
)

// Multiple subscribers get the same tap side effects
sub1 := source.Subscribe(ro.PrintObserver[int64]())
sub2 := source.Subscribe(ro.PrintObserver[int64]())

time.Sleep(350 * time.Millisecond)
sub1.Unsubscribe()
sub2.Unsubscribe()

// Each subscriber triggers the tap side effects
```

### With cleanup operations

```go
cleanup := func() {
    fmt.Println("Cleaning up resources...")
}

obs := ro.Pipe[string, string](
    ro.Just("data1", "data2"),
    ro.TapOnComplete(cleanup),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Cleaning up resources...
```