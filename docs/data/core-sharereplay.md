---
name: ShareReplay
slug: sharereplay
sourceRef: operator_connectable.go#L195
type: core
category: connectable
signatures:
  - "func ShareReplay[T any](bufferSize int)"
playUrl: https://go.dev/play/p/QmsDbChzRgu
variantHelpers:
  - core#connectable#sharereplay
similarHelpers:
  - core#connectable#share
  - core#connectable#sharereplaywithconfig
position: 20
---

Creates a shared Observable that replays a specified number of items to future subscribers.

```go
// Create source that emits values over time
source := ro.Pipe[int64, int64](
    ro.Interval(100 * time.Millisecond),
    ro.Take[int64](5),
    ro.ShareReplay[int64](2), // Cache last 2 values
)

// First subscriber
sub1 := source.Subscribe(ro.OnNext(func(value int64) {
    fmt.Printf("Sub1: %d\n", value)
}))

time.Sleep(350 * time.Millisecond) // Let first 3-4 values emit

// Second subscriber joins later and gets replayed values
sub2 := source.Subscribe(ro.OnNext(func(value int64) {
    fmt.Printf("Sub2: %d\n", value)
}))

time.Sleep(300 * time.Millisecond)
sub1.Unsubscribe()
sub2.Unsubscribe()

// 
// Sub1: 0
// Sub1: 1
// Sub1: 2
// Sub2: 1 (replayed from cache)
// Sub2: 2 (replayed from cache)
// Sub1: 3
// Sub2: 3
// Sub1: 4
// Sub2: 4
```

### With bufferSize 1 (latest value only)

```go
source := ro.Pipe[string, string](
    ro.Just("first", "second", "third", "fourth"),
    ro.ShareReplay[string](1), // Cache only latest value
)

sub1 := source.Subscribe(ro.PrintObserver[string]())
time.Sleep(50 * time.Millisecond)
sub1.Unsubscribe()

// Second subscriber gets only the last value
sub2 := source.Subscribe(ro.PrintObserver[string]())
time.Sleep(50 * time.Millisecond)
sub2.Unsubscribe()

// 
// First subscriber: first, second, third, fourth
// Second subscriber: fourth (only last value replayed)
```

### With bufferSize 0 (no replay, just sharing)

```go
source := ro.Pipe[int64, int64](
    ro.Interval(50 * time.Millisecond),
    ro.Take[int64](3),
    ro.ShareReplay[int64](0), // No replay, just sharing
)

sub1 := source.Subscribe(ro.OnNext(func(value int64) {
    fmt.Printf("Sub1: %d\n", value)
}))

time.Sleep(125 * time.Millisecond) // After 2-3 values

sub2 := source.Subscribe(ro.OnNext(func(value int64) {
    fmt.Printf("Sub2: %d\n", value)
}))

time.Sleep(100 * time.Millisecond)
sub1.Unsubscribe()
sub2.Unsubscribe()

// 
// Sub1: 0
// Sub1: 1
// Sub2: 2 (sub2 starts here, no replay)
// Sub1: 2
```

### With expensive operation caching

```go
// Simulate expensive API call
expensiveAPI := func() Observable[string] {
    return ro.Defer(func() Observable[string] {
        fmt.Println("🚀 Expensive API call...")
        time.Sleep(100 * time.Millisecond)
        return ro.Just("result1", "result2", "result3")
    })
}

// Cache the last 2 results
cachedAPI := ro.Pipe[string, string](
    expensiveAPI(),
    ro.ShareReplay[string](2),
)

// First subscriber triggers API call
sub1 := cachedAPI.Subscribe(ro.PrintObserver[string]())
time.Sleep(200 * time.Millisecond)
sub1.Unsubscribe()

// Subsequent subscribers get cached results (no new API call)
sub2 := cachedAPI.Subscribe(ro.PrintObserver[string]())
time.Sleep(50 * time.Millisecond)
sub2.Unsubscribe()

sub3 := cachedAPI.Subscribe(ro.PrintObserver[string]())
time.Sleep(50 * time.Millisecond)
sub3.Unsubscribe()

// 
// 🚀 Expensive API call...
// Sub1: result1, result2, result3
// Sub2: result2, result3 (from cache)
// Sub3: result2, result3 (from cache)
```

### With error handling

```go
source := ro.Pipe[int, int](
    Defer(func() Observable[int] {
        fmt.Println("Source execution...")
        return ro.Pipe[int, int](
            ro.Just(1, 2, 3),
            ro.MapErr(func(i int) (int, error) {
                if i == 3 {
                    return 0, errors.New("api failure")
                }
                return i, nil
            }),
        )
    }),
    ro.ShareReplay[int](2),
)

sub1 := source.Subscribe(ro.NewObserver(
    func(value int) {
        fmt.Printf("Sub1: %d\n", value)
    },
    func(err error) {
        fmt.Printf("Sub1 Error: %v\n", err)
    },
    func() {
        fmt.Println("Sub1 completion")
    },
))

time.Sleep(100 * time.Millisecond)

// Second subscriber gets replayed values before error
sub2 := source.Subscribe(ro.NewObserver(
    func(value int) {
        fmt.Printf("Sub2: %d\n", value)
    },
    func(err error) {
        fmt.Printf("Sub2 Error: %v\n", err)
    },
    func() {
        fmt.Println("Sub2 completion")
    },
))

time.Sleep(100 * time.Millisecond)
sub1.Unsubscribe()
sub2.Unsubscribe()

// 
// Source execution...
// Sub1: 1
// Sub1: 2
// Sub2: 1 (replayed)
// Sub2: 2 (replayed)
// Sub1 Error: api failure
// Sub2 Error: api failure
```

### With hot observable and late subscribers

```go
// Create a hot observable with replay
hotSource := ro.Pipe[int64, int64](
    ro.Interval(100 * time.Millisecond),
    ro.Take[int64](8),
    ro.ShareReplay[int64](3), // Cache last 3 values
)

// Simulate subscribers joining at different times
go func() {
    time.Sleep(0 * time.Millisecond)
    sub := hotSource.Subscribe(ro.OnNext(func(value int64) {
        fmt.Printf("Early sub: %d\n", value)
    }))
    time.Sleep(500 * time.Millisecond)
    sub.Unsubscribe()
}()

go func() {
    time.Sleep(250 * time.Millisecond)
    sub := hotSource.Subscribe(ro.OnNext(func(value int64) {
        fmt.Printf("Middle sub: %d\n", value)
    }))
    time.Sleep(400 * time.Millisecond)
    sub.Unsubscribe()
}()

go func() {
    time.Sleep(500 * time.Millisecond)
    sub := hotSource.Subscribe(ro.OnNext(func(value int64) {
        fmt.Printf("Late sub: %d\n", value)
    }))
    time.Sleep(400 * time.Millisecond)
    sub.Unsubscribe()
}()

time.Sleep(1200 * time.Millisecond)

// Shows replay behavior for late subscribers
```

### With large buffer for complete history

```go
source := ro.Pipe[string, string](
    ro.Just("apple", "banana", "cherry", "date", "elderberry"),
    ro.ShareReplay[string](10), // Large enough for all values
)

sub1 := source.Subscribe(ro.PrintObserver[string]())
time.Sleep(50 * time.Millisecond)
sub1.Unsubscribe()

// Subscribers joining later get complete history
sub2 := source.Subscribe(ro.PrintObserver[string]())
time.Sleep(50 * time.Millisecond)
sub2.Unsubscribe()

sub3 := source.Subscribe(ro.PrintObserver[string]())
time.Sleep(50 * time.Millisecond)
sub3.Unsubscribe()

// All subscribers get the complete sequence
// due to large replay buffer
```