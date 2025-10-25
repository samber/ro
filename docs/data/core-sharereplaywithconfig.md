---
name: ShareReplayWithConfig
slug: sharereplaywithconfig
sourceRef: operator_connectable.go#L220
type: core
category: connectable
signatures:
  - "func ShareReplayWithConfig[T any](bufferSize int, config ShareReplayConfig)"
playUrl: https://go.dev/play/p/rztZ6B4XiQL
variantHelpers:
  - core#connectable#sharereplaywithconfig
similarHelpers:
  - core#connectable#sharereplay
  - core#connectable#sharewithconfig
position: 30
---

Creates a shared Observable with replay functionality and custom configuration. Provides control over buffer size and reset behavior.

```go
config := ShareReplayConfig{
    ResetOnRefCountZero: true, // Reset when no subscribers left
}

source := ro.Pipe[string, string](
    ro.Just("first", "second", "third"),
    ro.ShareReplayWithConfig[string](2, config), // Cache last 2 values
)

sub1 := source.Subscribe(ro.PrintObserver[string]())
time.Sleep(50 * time.Millisecond)
sub1.Unsubscribe()

// Second subscriber gets replayed values
sub2 := source.Subscribe(ro.PrintObserver[string]())
time.Sleep(50 * time.Millisecond)
sub2.Unsubscribe()

// 
// Sub1: first, second, third
// Sub2: second, third (replayed from cache)
```

### With ResetOnRefCountZero disabled

```go
config := ShareReplayConfig{
    ResetOnRefCountZero: false, // Keep cache even when no subscribers
}

source := ro.Pipe[int, int](
    ro.Defer(func() ro.Observable[int] {
        fmt.Println("🔄 Creating new source...")
        return ro.Just(1, 2, 3, 4, 5)
    }),
    ro.ShareReplayWithConfig[int](3, config), // Cache last 3 values
)

// First subscriber triggers source creation
sub1 := source.Subscribe(ro.PrintObserver[int]())
time.Sleep(50 * time.Millisecond)
sub1.Unsubscribe()

// Wait a bit, then subscribe again
time.Sleep(100 * time.Millisecond)
fmt.Println("Subscribing again...")

// Second subscriber gets cached values (no new source creation)
sub2 := source.Subscribe(ro.PrintObserver[int]())
time.Sleep(50 * time.Millisecond)
sub2.Unsubscribe()

// 
// 🔄 Creating new source...
// Sub1: 1, 2, 3, 4, 5
// Subscribing again...
// Sub2: 3, 4, 5 (replayed from persistent cache)
```

### With ResetOnRefCountZero enabled

```go
config := ShareReplayConfig{
    ResetOnRefCountZero: true, // Reset cache when no subscribers
}

source := ro.Pipe[string, string](
    ro.Defer(func() ro.Observable[string] {
        fmt.Println("🔄 New source execution...")
        return ro.Just("hello", "world", "again")
    }),
    ro.ShareReplayWithConfig[string](2, config),
)

// First subscriber
sub1 := source.Subscribe(ro.PrintObserver[string]())
time.Sleep(50 * time.Millisecond)
sub1.Unsubscribe()

// Cache resets after all unsubscribe
time.Sleep(50 * time.Millisecond)

// Second subscriber triggers new source creation
sub2 := source.Subscribe(ro.PrintObserver[string]())
time.Sleep(50 * time.Millisecond)
sub2.Unsubscribe()

// 
// 🔄 New source execution...
// Sub1: hello, world, again
// 🔄 New source execution... (cache was reset)
// Sub2: hello, world, again
```

### With large buffer and persistent cache

```go
config := ShareReplayConfig{
    ResetOnRefCountZero: false, // Keep cache forever
}

source := ro.Pipe[string, string](
    ro.Just("data1", "data2", "data3", "data4", "data5"),
    ro.ShareReplayWithConfig[string](10, config), // Large buffer
)

sub1 := source.Subscribe(ro.PrintObserver[string]())
time.Sleep(50 * time.Millisecond)
sub1.Unsubscribe()

// Multiple subscribers can get complete history
for i := 2; i <= 4; i++ {
    sub := source.Subscribe(ro.PrintObserver[string]())
    time.Sleep(50 * time.Millisecond)
    sub.Unsubscribe()
}

// All subscribers get the complete sequence
// due to persistent large cache
```

### With expensive operation and persistent cache

```go
config := ShareReplayConfig{
    ResetOnRefCountZero: false, // Cache expensive results
}

expensiveOperation := func() ro.Observable[string] {
    return ro.Defer(func() ro.Observable[string] {
        fmt.Println("💸 Expensive database query...")
        time.Sleep(200 * time.Millisecond)
        return ro.Just("user1", "user2", "user3")
    })
}

// Cache the expensive operation results
cachedUsers := ro.Pipe[string, string](
    expensiveOperation(),
    ro.ShareReplayWithConfig[string](5, config),
)

// Multiple subscribers over time without re-querying
for i := 1; i <= 3; i++ {
    time.Sleep(300 * time.Millisecond)
    fmt.Printf("Query %d:\n", i)
    sub := cachedUsers.Subscribe(ro.PrintObserver[string]())
    time.Sleep(50 * time.Millisecond)
    sub.Unsubscribe()
}

// 
// 💸 Expensive database query...
// Query 1: user1, user2, user3
// Query 2: user1, user2, user3 (from cache)
// Query 3: user1, user2, user3 (from cache)
```

### With real-time data stream

```go
config := ShareReplayConfig{
    ResetOnRefCountZero: false, // Keep latest data available
}

// Simulate real-time price updates
priceStream := ro.Pipe[int64, float64](
    ro.Interval(1 * time.Second),
    ro.Map(func(_ int64) float64 {
        return 100 + rand.Float64()*10 // Price between 100-110
    }),
    ro.ShareReplayWithConfig[float64](1, config), // Keep only latest price
)

// Multiple price checkers
for i := 1; i <= 3; i++ {
    go func(checkerID int) {
        time.Sleep(time.Duration(checkerID) * 500 * time.Millisecond)
        sub := priceStream.Subscribe(ro.OnNext(func(price float64) {
            fmt.Printf("Checker %d: Price $%.2f\n", checkerID, price)
        }))
        time.Sleep(2 * time.Second)
        sub.Unsubscribe()
    }(i)
}

time.Sleep(4 * time.Second)

// Each checker gets the latest available price
// when they subscribe
```

### With error handling and persistent cache

```go
config := ShareReplayConfig{
    ResetOnRefCountZero: false, // Keep error state too
}

source := ro.Pipe[int, int](
    ro.Defer(func() ro.Observable[int] {
        fmt.Println("🔄 Attempting operation...")
        if rand.Intn(3) == 0 {
            return ro.Throw[int](errors.New("random failure"))
        }
        return ro.Just(42, 84, 126)
    }),
    ro.ShareReplayWithConfig[int](3, config),
)

// Multiple attempts may get cached error or success
for i := 1; i <= 3; i++ {
    time.Sleep(200 * time.Millisecond)
    fmt.Printf("Attempt %d:\n", i)
    sub := source.Subscribe(ro.NewObserver(
        func(value int) {
            fmt.Printf("  Success: %d\n", value)
        },
        func(err error) {
            fmt.Printf("  Error: %v\n", err)
        },
        func() {
            fmt.Println("  Completed")
        },
    ))
    time.Sleep(50 * time.Millisecond)
    sub.Unsubscribe()
}

// If first attempt fails, subsequent attempts get cached error
// If first succeeds, subsequent attempts get cached success
```

### With buffer management

```go
config := ShareReplayConfig{
    ResetOnRefCountZero: true,
}

// Stream with varying data rates
dataStream := ro.Pipe[int64, int64](
    ro.Interval(100 * time.Millisecond),
    ro.Take[int64](20),
    ro.ShareReplayWithConfig[int64](5, config), // Keep last 5 values
)

// Simulate periodic subscribers
for i := 0; i < 4; i++ {
    go func(batch int) {
        time.Sleep(time.Duration(batch) * 300 * time.Millisecond)
        fmt.Printf("Batch %d subscribing:\n", batch+1)
        sub := dataStream.Subscribe(ro.OnNext(func(value int64) {
            fmt.Printf("  B%d: %d\n", batch+1, value)
        }))
        time.Sleep(400 * time.Millisecond)
        sub.Unsubscribe()
        fmt.Printf("Batch %d done\n", batch+1)
    }(i)
}

time.Sleep(1500 * time.Millisecond)

// Shows how each batch gets replayed last 5 values
// from when they subscribed
```