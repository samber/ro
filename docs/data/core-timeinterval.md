---
name: TimeInterval
slug: timeinterval
sourceRef: operator_utility.go#L245
type: core
category: utility
signatures:
  - "func TimeInterval[T any]()"
playUrl: https://go.dev/play/p/VX73ZL74hPk
variantHelpers:
  - core#utility#timeinterval
similarHelpers:
  - core#utility#timestamp
position: 200
---

Records the interval of time between emissions from the source Observable and emits this information as ro.IntervalValue objects.

```go
obs := ro.Pipe[string, ro.IntervalValue[string]](
    ro.Just("A", "B", "C"),
    ro.TimeInterval[string](),
)

sub := obs.Subscribe(ro.PrintObserver[ro.IntervalValue[string]]())
defer sub.Unsubscribe()

// Shows interval between emissions:
// Next: {Value:A Interval:0ms}
// Next: {Value:B Interval:<time between A and B>}
// Next: {Value:C Interval:<time between B and C>}
// Completed
```

### With hot observable

```go
obs := ro.Pipe[int64, ro.IntervalValue[int64]](
    ro.Interval(100 * time.Millisecond),
    ro.Take[int64](5),
    ro.TimeInterval[int64](),
)

sub := obs.Subscribe(ro.OnNext(func(value ro.IntervalValue[int64]) {
    fmt.Printf("Value: %d, Interval: %v\n", value.Value, value.Interval)
))
time.Sleep(700 * time.Millisecond)
sub.Unsubscribe()

// 
// Value: 0, Interval: 0s
// Value: 1, Interval: ~100ms
// Value: 2, Interval: ~100ms
// Value: 3, Interval: ~100ms
// Value: 4, Interval: ~100ms
```

### With async operations

```go
obs := ro.Pipe[string, ro.IntervalValue[string]](
    ro.Pipe[string, string](
        ro.Just("task1", "task2", "task3"),
        ro.MapAsync(func(task string) Observable[string] {
            return ro.Defer(func() Observable[string] {
                delay := time.Duration(rand.Intn(200)) * time.Millisecond
                time.Sleep(delay)
                return ro.Just(task)
            })
        }, 2),
    ),
    ro.TimeInterval[string](),
)

sub := obs.Subscribe(ro.OnNext(func(value ro.IntervalValue[string]) {
    fmt.Printf("Task: %s completed after %v\n", value.Value, value.Interval)
}))
time.Sleep(500 * time.Millisecond)
defer sub.Unsubscribe()

// Shows variable intervals due to async processing
```

### With error handling

```go
obs := ro.Pipe[int, ro.IntervalValue[int]](
    ro.Pipe[int, int](
        ro.Just(1, 2, 3),
        ro.MapErr(func(i int) (int, error) {
            if i == 3 {
                return 0, fmt.Errorf("error on 3")
            }
            time.Sleep(50 * time.Millisecond)
            return i, nil
        }),
    ),
    ro.TimeInterval[int](),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value ro.IntervalValue[int]) {
        fmt.Printf("Value: %d, Interval: %v\n", value.Value, value.Interval)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Complete")
    },
))
defer sub.Unsubscribe()

// 
// Value: 1, Interval: 0s
// Value: 2, Interval: ~50ms
// Error: error on 3
```

### With performance monitoring

```go
// Monitor processing time for expensive operations
obs := ro.Pipe[string, ro.IntervalValue[string]](
    ro.Just("data1", "data2", "data3"),
    ro.Map(func(data string) string {
        // Simulate expensive processing
        time.Sleep(100 * time.Millisecond)
        return "processed_" + data
    }),
    ro.TimeInterval[string](),
)

sub := obs.Subscribe(ro.OnNext(func(value ro.IntervalValue[string]) {
    fmt.Printf("Processed: %s in %v\n", value.Value, value.Interval)
}))
defer sub.Unsubscribe()

// 
// Processed: processed_data1 in ~100ms
// Processed: processed_data2 in ~100ms
// Processed: processed_data3 in ~100ms
```

### With real-time data stream

```go
// Monitor intervals in real-time data stream
source := ro.Interval(200 * time.Millisecond)
obs := ro.Pipe[float64, ro.IntervalValue[float64]](
    source,
    ro.Take[int64](10),
    ro.Map(func(ts int64) float64 {
        // Simulate sensor reading
        return 20.0 + rand.Float64()*10
    }),
    ro.TimeInterval[float64](),
)

sub := obs.Subscribe(ro.OnNext(func(value ro.IntervalValue[float64]) {
    fmt.Printf("[%v] Reading: %.2f (interval: %v)\n",
        time.Now().Format("15:04:05.000"),
        value.Value,
        value.Interval.Round(time.Millisecond))
}))
time.Sleep(2500 * time.Millisecond)
sub.Unsubscribe()
```

### With batch processing analysis

```go
// Analyze batch processing times
obs := ro.Pipe[[]int, ro.IntervalValue[[]int]](
    ro.Range(1, 6),
    ro.BufferWithCount[int](2),
    ro.TimeInterval[[]int](),
)

sub := obs.Subscribe(ro.OnNext(func(value ro.IntervalValue[[]int]) {
    fmt.Printf("Batch %v processed in %v\n", value.Value, value.Interval)
}))
defer sub.Unsubscribe()

// 
// Batch [1 2] processed in ~0s
// Batch [3 4] processed in ~<interval>
// Batch [5] processed in ~<interval>
```

### With conditional intervals

```go
// Measure intervals only for certain values
obs := ro.Pipe[int, ro.IntervalValue[int]](
    ro.Range(1, 10),
    ro.Filter(func(i int) bool {
        return i%3 == 0 // Only multiples of 3
    }),
    ro.TimeInterval[int](),
)

sub := obs.Subscribe(ro.OnNext(func(value ro.IntervalValue[int]) {
        fmt.Printf("Filtered value: %d (interval: %v)\n", value.Value, value.Interval)
}))
defer sub.Unsubscribe()

// 
// Filtered value: 3 (interval: <time from start to 3>)
// Filtered value: 6 (interval: <time between 3 and 6>)
// Filtered value: 9 (interval: <time between 6 and 9>)
```