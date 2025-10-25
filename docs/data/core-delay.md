---
name: Delay
slug: delay
sourceRef: operator_utility.go#L278
type: core
category: utility
signatures:
  - "func Delay[T any](duration time.Duration)"
playUrl: https://go.dev/play/p/BQBrPN7Fj6R
variantHelpers:
  - core#utility#delay
similarHelpers:
  - core#utility#delayeach
  - core#utility#timeout
position: 220
---

Shifts the emissions from the source Observable forward in time by a specified amount.

```go
obs := ro.Pipe[string, string](
    ro.Just("A", "B", "C"),
    ro.Delay(100 * time.Millisecond),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(200 * time.Millisecond)
defer sub.Unsubscribe()

// (100ms delay)
// Next: A
// Next: B
// Next: C
// Completed
```

### With hot observable

```go
start := time.Now()
obs := ro.Pipe[int64, int64](
    ro.Interval(50 * time.Millisecond),
    ro.Take[int64](3),
    ro.Delay(200 * time.Millisecond),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value int64) {
        elapsed := time.Since(start)
        fmt.Printf("Value: %d at %v\n", value, elapsed.Round(time.Millisecond))
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
time.Sleep(500 * time.Millisecond)
sub.Unsubscribe()

// Value: 0 at ~250ms (200ms delay + 50ms emission)
// Value: 1 at ~300ms
// Value: 2 at ~350ms
```

### With error propagation

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.MapErr(func(i int) (int, error) {
        if i == 3 {
            return 0, fmt.Errorf("error on 3")
        }
        return i, nil
    }),
    ro.Delay(100 * time.Millisecond),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value int) {
        fmt.Printf("Next: %d\n", value)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
time.Sleep(300 * time.Millisecond)
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Error: error on 3 (delayed by 100ms)
```

### With multiple delays in pipeline

```go
obs := ro.Pipe[string, string](
    ro.Just("start", "middle", "end"),
    ro.Delay(50 * time.Millisecond),
    ro.Map(func(s string) string {
        return s + "_processed"
    }),
    ro.Delay(25 * time.Millisecond),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value string) {
        fmt.Printf("Received: %s\n", value)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
time.Sleep(200 * time.Millisecond)
defer sub.Unsubscribe()

// Total delay: ~75ms per item
// Received: start_processed
// Received: middle_processed
// Received: end_processed
```

### With async operations

```go
obs := ro.Pipe[string, string](
    ro.Just("task1", "task2", "task3"),
    ro.MapAsync(func(task string) ro.Observable[string] {
        return ro.Defer(func() ro.Observable[string] {
            time.Sleep(30 * time.Millisecond)
            return ro.Just(task + "_done")
        })
    }, 2),
    ro.Delay(100 * time.Millisecond),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value string) {
        fmt.Printf("Task completed: %s\n", value)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
time.Sleep(400 * time.Millisecond)
defer sub.Unsubscribe()

// Each async result is delayed by additional 100ms
```

### With context cancellation

```go
ctx, cancel := context.WithTimeout(context.Background(), 200 * time.Millisecond)
defer cancel()

obs := ro.Pipe[int64, int64](
    ro.Interval(50 * time.Millisecond),
    ro.Delay(150 * time.Millisecond),
)

sub := obs.SubscribeWithContext(ctx, ro.NewObserver(
    func(value int64) {
        fmt.Printf("Value: %d\n", value)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))

// No values will be emitted because context times out
// before the delay completes
time.Sleep(300 * time.Millisecond)
sub.Unsubscribe()

// Error: context deadline exceeded
```

### With real-time data

```go
// Delay real-time price updates
type PriceUpdate struct {
    Symbol string
    Price  float64
    Time   time.Time
}

obs := ro.Pipe[int64, PriceUpdate](
    ro.Interval(100 * time.Millisecond),
    ro.Take[int64](5),
    ro.Map(func(ts int64) PriceUpdate {
        return PriceUpdate{
            Symbol: "BTC",
            Price:  50000 + rand.Float64()*1000,
            Time:   time.Now(),
        }
    }),
    ro.Delay(200 * time.Millisecond),
)

sub := obs.Subscribe(ro.NewObserver(
    func(update PriceUpdate) {
        delay := time.Since(update.Time)
        fmt.Printf("Price: $%.2f (delayed by %v)\n", update.Price, delay.Round(time.Millisecond))
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
time.Sleep(1 * time.Second)
sub.Unsubscribe()
```

### With conditional application

```go
// Apply delay only to certain items
obs := ro.Pipe[string, string](
    ro.Just(
        "immediate", // No delay
        "delayed",   // Apply delay
        "immediate", // No delay
    ),
    ro.Map(func(item string) ro.Observable[string] {
        if item == "delayed" {
            return ro.Pipe[string, string](
                ro.Just(item),
                ro.Delay(100 * time.Millisecond),
            )
        }
        return ro.Just(item)
    }),
    ro.Merge[string](),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value string) {
        fmt.Printf("Received: %s\n", value)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
time.Sleep(200 * time.Millisecond)
defer sub.Unsubscribe()

// Received: immediate
// Received: immediate
// (100ms delay)
// Received: delayed
```