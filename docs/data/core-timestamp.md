---
name: Timestamp
slug: timestamp
sourceRef: operator_utility.go#L258
type: core
category: utility
signatures:
  - "func Timestamp[T any]()"
playUrl: https://go.dev/play/p/cDiCr6qIE2P
variantHelpers:
  - core#utility#timestamp
similarHelpers:
  - core#utility#timeinterval
position: 210
---

Attaches a timestamp to each emission from the source Observable, indicating when it was emitted.

```go
obs := ro.Pipe[string, TimestampValue[string]](
    ro.Just("A", "B", "C"),
    ro.Timestamp[string](),
)

sub := obs.Subscribe(ro.PrintObserver[TimestampValue[string]]())
defer sub.Unsubscribe()

// Next: {Value:A Timestamp:<current time>}
// Next: {Value:B Timestamp:<current time>}
// Next: {Value:C Timestamp:<current time>}
// Completed
```

### With hot observable

```go
obs := ro.Pipe[int64, TimestampValue[int64]](
    ro.Interval(100 * time.Millisecond),
    ro.Take[int64](3),
    ro.Timestamp[int64](),
)

sub := obs.Subscribe(ro.OnNext(
    func(value TimestampValue[int64]) {
        fmt.Printf("Value: %d at %v\n", value.Value, value.Timestamp.Format("15:04:05.000"))
    },
))
time.Sleep(500 * time.Millisecond)
sub.Unsubscribe()

// Value: 0 at 12:34:56.789
// Value: 1 at 12:34:56.889
// Value: 2 at 12:34:56.989
```

### With async operations

```go
obs := ro.Pipe[string, TimestampValue[string]](
    ro.Pipe[string, TimestampValue[string]](
        ro.Just("task1", "task2", "task3"),
        ro.MapAsync(func(task string) Observable[string] {
            return ro.Defer(func() Observable[string] {
                time.Sleep(50 * time.Millisecond)
                return ro.Just(task)
            })
        }, 2),
    ),
    ro.Timestamp[string](),
)

sub := obs.Subscribe(ro.OnNext(func(value TimestampValue[string]) {
    fmt.Printf("Task %s completed at %v\n", value.Value, value.Timestamp.Format("15:04:05.000"))
}))
time.Sleep(300 * time.Millisecond)
defer sub.Unsubscribe()
```

### With data logging

```go
type LogEntry struct {
    Message   string
    Level     string
    Timestamp time.Time
}

obs := ro.Pipe[LogEntry, TimestampValue[LogEntry]](
    ro.Just(
        LogEntry{Message: "Server started", Level: "INFO"},
        LogEntry{Message: "User connected", Level: "INFO"},
        LogEntry{Message: "Database error", Level: "ERROR"},
    ),
    ro.Timestamp[LogEntry](),
)

sub := obs.Subscribe(ro.OnNext(func(value TimestampValue[LogEntry]) {
    entry := value.Value
    fmt.Printf("[%s] %s: %s\n",
        value.Timestamp.Format("2006-01-02 15:04:05"),
        entry.Level,
        entry.Message)
}))
defer sub.Unsubscribe()

// [2024-01-01 12:34:56] INFO: Server started
// [2024-01-01 12:34:56] INFO: User connected
// [2024-01-01 12:34:56] ERROR: Database error
```

### With real-time sensor data

```go
type SensorReading struct {
    ID        string
    Value     float64
    Timestamp time.Time
}

obs := ro.Pipe[int64, TimestampValue[SensorReading]](
    ro.Interval(1 * time.Second),
    ro.Take[int64](5),
    ro.Map(func(ts int64) SensorReading {
        return SensorReading{
            ID:        "temp-01",
            Value:     20.0 + rand.Float64()*10,
            Timestamp: time.Now(),
        }
    }),
    ro.Timestamp[SensorReading](),
)

sub := obs.Subscribe(ro.OnNext(
    func(value TimestampValue[SensorReading]) {
        reading := value.Value
        fmt.Printf("[%s] %s: %.2f°C (system time: %v)\n",
            reading.Timestamp.Format("15:04:05"),
            reading.ID,
            reading.Value,
            value.Timestamp.Format("15:04:05.000"))
    },
))
time.Sleep(6 * time.Second)
sub.Unsubscribe()
```

### With event ordering

```go
// Track event order and timing
type Event struct {
    ID      string
    Action  string
    Payload interface{}
}

obs := ro.Pipe[Event, TimestampValue[Event]](
    ro.Just(
        Event{ID: "1", Action: "click", Payload: "button"},
        Event{ID: "2", Action: "scroll", Payload: 100},
        Event{ID: "3", Action: "input", Payload: "text"},
    ),
    ro.Timestamp[Event](),
)

sub := obs.Subscribe(ro.OnNext(func(value TimestampValue[Event]) {
    event := value.Value
    fmt.Printf("%s | Event %s: %s (%v)\n",
        value.Timestamp.Format("15:04:05.000"),
        event.ID,
        event.Action,
        event.Payload)
}))
defer sub.Unsubscribe()

// Shows precise timing of each event
```

### With batch processing

```go
// Timestamp batch completion times
obs := ro.Pipe[int64, TimestampValue[[]int]](
    ro.Range(1, 10),
    ro.BufferWithCount[int](3),
    ro.Timestamp[[]int](),
)

sub := obs.Subscribe(ro.OnNext(func(value TimestampValue[[]int]) {
    fmt.Printf("Batch %v completed at %v\n",
        value.Value,
        value.Timestamp.Format("15:04:05.000"))
}))
defer sub.Unsubscribe()

// Batch [1 2 3] completed at 12:34:56.789
// Batch [4 5 6] completed at 12:34:56.789
// Batch [7 8 9] completed at 12:34:56.789
```

### With error tracking

```go
obs := ro.Pipe[int, TimestampValue[int]](
    ro.Pipe[int, int](
        ro.Just(1, 2, 3),
        ro.MapErr(func(i int) (int, error) {
            if i == 3 {
                return 0, fmt.Errorf("processing error")
            }
            return i * 10, nil
        }),
    ),
    ro.Timestamp[int](),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value TimestampValue[int]) {
        fmt.Printf("Success: %d at %v\n", value.Value, value.Timestamp.Format("15:04:05.000"))
    },
    func(err error) {
        fmt.Printf("Error occurred at %v: %v\n", time.Now().Format("15:04:05.000"), err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Success: 10 at 12:34:56.789
// Success: 20 at 12:34:56.789
// Error occurred at 12:34:56.789: processing error
```
