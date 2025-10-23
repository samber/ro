---
name: ToSlice
slug: toslice
sourceRef: operator_sink.go#L30
type: core
category: sink
signatures:
  - "func ToSlice[T any]()"
playUrl:
variantHelpers:
  - core#sink#toslice
similarHelpers:
  - core#sink#tomap
  - core#sink#tochannel
position: 10
---

Collects all emissions from the source Observable into a single slice and emits that slice when the source completes.

```go
obs := ro.Pipe[int, []int](
    ro.Just(1, 2, 3, 4, 5),
    ro.ToSlice[int](),
)

sub := obs.Subscribe(ro.PrintObserver[[]int]())
defer sub.Unsubscribe()

// Next: [1 2 3 4 5]
// Completed
```

### With empty observable

```go
obs := ro.Pipe[int, []int](
    ro.Empty[int](),
    ro.ToSlice[int](),
)

sub := obs.Subscribe(ro.PrintObserver[[]int]())
defer sub.Unsubscribe()

// Next: [] (empty slice)
// Completed
```

### With error handling

```go
obs := ro.Pipe[int, []int](
    ro.Pipe[int, int](
        ro.Just(1, 2, 3),
        ro.MapErr(func(i int) (int, error) {
            if i == 3 {
                return 0, fmt.Errorf("error on 3")
            }
            return i, nil
        }),
    ),
    ro.ToSlice[int](),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value []int) {
        fmt.Printf("Slice: %v\n", value)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Complete")
    },
))
defer sub.Unsubscribe()

// Error: error on 3
// (No slice emitted due to error)
```

### With hot observable

```go
source := ro.Interval(100 * time.Millisecond)
obs := ro.Pipe[int64, []int64](
    source,
    ro.Take[int64](5),
    ro.ToSlice[int64](),
)

sub := obs.Subscribe(ro.PrintObserver[[]int64]())
time.Sleep(700 * time.Millisecond)
sub.Unsubscribe()

// Next: [0 1 2 3 4]
// Completed
```

### With large data sets

```go
obs := ro.Pipe[int, []int](
    ro.Range(1, 1000),
    ro.ToSlice[int](),
)

sub := obs.Subscribe(ro.OnNext(func(value []int) {
    fmt.Printf("Collected %d items\n", len(value))
    fmt.Printf("First: %d, Last: %d\n", value[0], value[len(value)-1])
}))
defer sub.Unsubscribe()

// Collected 1000 items
// First: 1, Last: 1000
```

### With conditional emission

```go
obs := ro.Pipe[int, []int](
    ro.Range(1, 10),
    ro.Filter(func(i int) bool {
        return i%2 == 0 // Only even numbers
    }),
    ro.ToSlice[int](),
)

sub := obs.Subscribe(ro.PrintObserver[[]int]())
defer sub.Unsubscribe()

// Next: [2 4 6 8]
// Completed
```

### With transformation pipeline

```go
obs := ro.Pipe[string, []int](
    ro.Just("hello", "world", "reactive", "programming"),
    ro.Map(func(s string) int {
        return len(s)
    }),
    ro.Filter(func(length int) bool {
        return length > 4
    }),
    ro.ToSlice[int](),
)

sub := obs.Subscribe(ro.PrintObserver[[]int]())
defer sub.Unsubscribe()

// Next: [5 5 11]
// Completed
```

### With async operations

```go
obs := ro.Pipe[string, []string](
    ro.Pipe[string, string](
        ro.Just("url1", "url2", "url3"),
        ro.MapAsync(func(url string) ro.Observable[string] {
            return ro.Defer(func() ro.Observable[string] {
                time.Sleep(50 * time.Millisecond)
                return ro.Just("data_" + url)
            })
        }, 2),
    ),
    ro.ToSlice[string](),
)

sub := obs.Subscribe(ro.PrintObserver[[]string]())
time.Sleep(300 * time.Millisecond)
defer sub.Unsubscribe()

// Next: [data_url1 data_url2 data_url3]
// Completed
```

### With real-time data collection

```go
// Simulate sensor readings
sensorData := ro.Interval(1 * time.Second)
obs := ro.Pipe[int64, []float64](
    sensorData,
    ro.Take[int64](10),
    ro.Map(func(timestamp int64) float64 {
        // Simulate temperature readings
        return 20.0 + rand.Float64()*10
    }),
    ro.ToSlice[float64](),
)

sub := obs.Subscribe(ro.OnNext(func(readings []float64) {
    fmt.Printf("Collected %d temperature readings\n", len(readings))
    avg := 0.0
    for _, r := range readings {
        avg += r
    }
    avg /= float64(len(readings))
    fmt.Printf("Average temperature: %.2f°C\n", avg)
}))
time.Sleep(12 * time.Second)
sub.Unsubscribe()
```

### With batch processing

```go
// Process items in batches but collect results
obs := ro.Pipe[int, []string](
    ro.Range(1, 25),
    ro.BufferWithCount[int](5),
    ro.Map(func(batch []int) string {
        sum := 0
        for _, num := range batch {
            sum += num
        }
        return fmt.Sprintf("batch-sum-%d", sum)
    }),
    ro.ToSlice[string](),
)

sub := obs.Subscribe(ro.PrintObserver[[]string]())
defer sub.Unsubscribe()

// Next: [batch-sum-15 batch-sum-40 batch-sum-65 batch-sum-90 batch-sum-115]
// Completed
```