---
name: ToChannel
slug: tochannel
sourceRef: operator_sink.go#L120
type: core
category: sink
signatures:
  - "func ToChannel[T any](bufferSize int)"
  - "func ToChannelWithContext[T any](ctx context.Context, bufferSize int)"
playUrl:
variantHelpers:
  - core#sink#tochannel
  - core#sink#tochannelwithcontext
similarHelpers:
  - core#sink#toslice
  - core#sink#tomap
position: 30
---

Converts the source Observable to a channel, emitting all values from the Observable through the channel.

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.ToChannel[int](0), // Unbuffered channel
)

sub := obs.Subscribe(ro.PrintObserver[<-chan int]())
defer sub.Unsubscribe()

// Next: 0xc0000a2000 (channel address)
// Completed

// You would typically consume the channel like this:
// channel := <-sub.Next()
// for value := range channel {
//     fmt.Println("Received:", value)
// }
```

### With buffered channel

```go
obs := ro.Pipe[string, string](
    ro.Just("hello", "world", "reactive"),
    ro.ToChannel[string](5), // Buffered channel with capacity 5
)

sub := obs.Subscribe(ro.OnNext(func(ch <-chan string) {
    fmt.Println("Channel received, consuming values:")
    for value := range ch {
        fmt.Printf("  %s\n", value)
    }
    fmt.Println("Channel closed")
}))
defer sub.Unsubscribe()

// Channel received, consuming values:
//   hello
//   world
//   reactive
// Channel closed
```

### ToChannelWithContext

```go
ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
defer cancel()

obs := ro.Pipe[int64, int64](
    ro.Interval(500*time.Millisecond),
    ro.ToChannelWithContext[int64](ctx, 3),
)

sub := obs.Subscribe(ro.OnNext(func(ch <-chan int64) {
    fmt.Println("Reading from channel with context:")
    for value := range ch {
        fmt.Printf("  Value: %d\n", value)
    }
    fmt.Println("Channel closed (context cancelled or source completed)")
}))
time.Sleep(3 * time.Second)
sub.Unsubscribe()

// Reading from channel with context:
//   Value: 0
//   Value: 1
//   Value: 2
//   Value: 3
// Channel closed (context cancelled or source completed)
```

### With hot observable and multiple consumers

```go
// Create a shared channel from a hot observable
source := ro.Interval(200 * time.Millisecond)

channelObs := ro.Pipe[int64, int64](
    source,
    ro.Take[int64](10),
    ro.ToChannel[int64](5),
)

sub := channelObs.Subscribe(ro.PrintObserver[<-chan int64]())
defer sub.Unsubscribe()

// Get the channel
var resultChan <-chan int64
sub = channelObs.Subscribe(ro.OnNext(func(ch <-chan int64) {
    resultChan = ch
}))
time.Sleep(100 * time.Millisecond)
sub.Unsubscribe()

if resultChan != nil {
    // Multiple goroutines can consume from the same channel
    var wg sync.WaitGroup

    // Consumer 1
    wg.Add(1)
    go func() {
        defer wg.Done()
        for i := 0; i < 5; i++ {
            value, ok := <-resultChan
            if !ok {
                break
            }
            fmt.Printf("Consumer 1: %d\n", value)
        }
    }()

    // Consumer 2
    wg.Add(1)
    go func() {
        defer wg.Done()
        for i := 0; i < 5; i++ {
            value, ok := <-resultChan
            if !ok {
                break
            }
            fmt.Printf("Consumer 2: %d\n", value)
        }
    }()

    wg.Wait()
}
```

### With error handling

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
    ro.ToChannel[int](3),
)

sub := obs.Subscribe(ro.NewObserver(
    func(ch <-chan int) {
        fmt.Println("Channel received, consuming values:")
        for value := range ch {
            fmt.Printf("  %d\n", value)
        }
        fmt.Println("Channel closed")
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
// (No channel emitted due to error)
```

### With finite stream and timeout

```go
obs := ro.Pipe[string, string](
    ro.Just("data1", "data2", "data3", "data4", "data5"),
    ro.ToChannel[string](2),
)

sub := obs.Subscribe(ro.OnNext(func(ch <-chan string) {
    fmt.Println("Processing channel with timeout:")

    timeout := time.After(3 * time.Second)
    for {
        select {
        case value, ok := <-ch:
            if !ok {
                fmt.Println("Channel closed normally")
                return
            }
            fmt.Printf("  Received: %s\n", value)
        case <-timeout:
            fmt.Println("Timeout reached")
            return
        }
    }
}))
defer sub.Unsubscribe()
```

### With complex data transformation

```go
type Event struct {
    ID      string
    Type    string
    Payload interface{}
    Time    time.Time
}

obs := ro.Pipe[Event, Event](
    ro.Just(
        Event{ID: "1", Type: "click", Payload: "button", Time: time.Now()},
        Event{ID: "2", Type: "scroll", Payload: 100, Time: time.Now()},
        Event{ID: "3", Type: "click", Payload: "link", Time: time.Now()},
    ),
    ro.ToChannel[Event](3),
)

sub := obs.Subscribe(ro.OnNext(func(ch <-chan Event) {
    fmt.Println("Processing events from channel:")
    for event := range ch {
        fmt.Printf("  Event %s (%s): %v\n", event.ID, event.Type, event.Payload)
    }
    fmt.Println("All events processed")
}))
defer sub.Unsubscribe()
```

### With backpressure control

```go
// Fast producer with slow consumer through buffered channel
fastProducer := ro.Interval(10 * time.Millisecond)  // 100 values/second
obs := ro.Pipe[int64, int64](
    fastProducer,
    ro.Take[int64](50),
    ro.ToChannel[int64](10), // Buffer of 10 provides backpressure
)

sub := obs.Subscribe(ro.OnNext(func(ch <-chan int64) {
    fmt.Println("Processing with backpressure:")
    processed := 0
    for value := range ch {
        // Simulate slow processing
        time.Sleep(50 * time.Millisecond)
        processed++
        if processed%10 == 0 {
            fmt.Printf("  Processed %d values (buffer managing backpressure)\n", processed)
        }
    }
    fmt.Printf("Total processed: %d\n", processed)
}))
time.Sleep(3 * time.Second)
sub.Unsubscribe()
```

### With context cancellation

```go
ctx, cancel := context.WithCancel(context.Background())

obs := ro.Pipe[int64, int64](
    ro.Interval(100 * time.Millisecond),
    ro.ToChannelWithContext[int64](ctx, 5),
)

sub := obs.Subscribe(ro.OnNext(func(ch <-chan int64) {
    fmt.Println("Reading from cancellable channel:")
    count := 0

    // Cancel after receiving 3 values
    go func() {
        time.Sleep(350 * time.Millisecond)
        fmt.Println("Cancelling context...")
        cancel()
    }()

    for value := range ch {
        count++
        fmt.Printf("  Value %d: %d\n", count, value)
    }
    fmt.Printf("Channel closed after %d values\n", count)
}))
time.Sleep(1 * time.Second)
sub.Unsubscribe()

// Reading from cancellable channel:
//   Value 1: 0
//   Value 2: 1
//   Value 3: 2
// Cancelling context...
// Channel closed after 3 values
```

### With real-time data streaming

```go
type Event struct {
    SensorID    string
    Temperature float64
    Humidity    float64
    Timestamp   time.Time
}

// Simulate real-time sensor data streaming
sensorData := func() Observable[Event] {
    return ro.Pipe[int64, Event](
        ro.Interval(1 * time.Second),
        ro.Map(func(_ int64) Event {
            return Event{
                SensorID:    "sensor-01",
                Temperature: 20.0 + rand.Float64()*10,
                Humidity:    40.0 + rand.Float64()*20,
                Timestamp:   time.Now(),
            }
        }),
    )
}

obs := ro.Pipe[Event, Event](
    sensorData(),
    ro.Take[Event](5),
    ro.ToChannel[Event](1),
)

sub := obs.Subscribe(ro.OnNext(
    func(ch <-chan Event) {
        fmt.Println("Real-time sensor data streaming:")
        for reading := range ch {
            fmt.Printf("  [%s] %s: %.1f°C, %.1f%%\n",
                reading.Timestamp.Format("15:04:05"),
                reading.SensorID,
                reading.Temperature,
                reading.Humidity)
        }
        fmt.Println("Stream ended")
    },
))
time.Sleep(6 * time.Second)
sub.Unsubscribe()
```