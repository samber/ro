---
name: Share
slug: share
sourceRef: operator_connectable.go#L38
type: core
category: connectable
signatures:
  - "func Share[T any]()"
playUrl:
variantHelpers:
  - core#connectable#share
similarHelpers:
  - core#connectable#sharewithconfig
  - core#connectable#sharereplay
position: 0
---

Creates a new Observable that multicasts (shares) the original Observable. This allows multiple subscribers to share the same underlying subscription.

```go
// Without Share - each subscriber gets separate execution
source := ro.Interval(100 * time.Millisecond).Take[int64](5)

obs1 := source
obs2 := source

sub1 := obs1.Subscribe(ro.OnNext(func(value int64) {
    fmt.Printf("Sub1: %d\n", value)
}))

sub2 := obs2.Subscribe(ro.OnNext(func(value int64) {
    fmt.Printf("Sub2: %d\n", value)
}))

time.Sleep(600 * time.Millisecond)
sub1.Unsubscribe()
sub2.Unsubscribe()

// 
// Sub1: 0
// Sub2: 0
// Sub1: 1
// Sub2: 1
// ... (each subscriber gets all values independently)
```

### With Share - shared execution

```go
// With Share - subscribers share the same execution
source := ro.Pipe[int64, int64](
    ro.Interval(100 * time.Millisecond),
    ro.Take[int64](5),
    ro.Share[int64](), // Share the observable
)

sub1 := source.Subscribe(ro.OnNext(func(value int64) {
    fmt.Printf("Shared Sub1: %d\n", value)
}))

// Second subscriber subscribes later
time.Sleep(250 * time.Millisecond)
sub2 := source.Subscribe(ro.OnNext(func(value int64) {
    fmt.Printf("Shared Sub2: %d\n", value)
}))

time.Sleep(500 * time.Millisecond)
sub1.Unsubscribe()
sub2.Unsubscribe()

// 
// Shared Sub1: 0
// Shared Sub1: 1
// Shared Sub2: 2  (sub2 starts here)
// Shared Sub1: 2
// Shared Sub2: 3
// Shared Sub1: 3
// Shared Sub2: 4
// Shared Sub1: 4
// (both subscribers share the same sequence)
```

### With expensive operations

```go
// Simulate expensive API call
expensiveOperation := func() Observable[string] {
    return ro.Defer(func() Observable[string] {
        fmt.Println("Expensive API call started...")
        time.Sleep(100 * time.Millisecond)
        return ro.Just("api_result_1", "api_result_2")
    })
}

// Without Share - each subscriber triggers separate API call
withoutShare := expensiveOperation()

sub1 := withoutShare.Subscribe(ro.PrintObserver[string]())
sub2 := withoutShare.Subscribe(ro.PrintObserver[string]())

time.Sleep(200 * time.Millisecond)
sub1.Unsubscribe()
sub2.Unsubscribe()

// 
// Expensive API call started...
// Expensive API call started...
// Next: api_result_1, Next: api_result_2 (twice)

// With Share - API call shared
withShare := ro.Pipe[string, string](
    expensiveOperation(),
    Share[string](),
)

sub3 := withShare.Subscribe(ro.PrintObserver[string]())
sub4 := withShare.Subscribe(ro.PrintObserver[string]())

time.Sleep(200 * time.Millisecond)
sub3.Unsubscribe()
sub4.Unsubscribe()

// 
// Expensive API call started...
// Next: api_result_1, Next: api_result_2 (once, shared)
```

### With error handling

```go
source := ro.Pipe[int, int](
    Defer(func() Observable[int] {
        fmt.Println("Source execution started...")
        return ro.Pipe[int, int](
            ro.Just(1, 2, 3),
            ro.MapErr(func(i int) (int, error) {
                if i == 3 {
                    return 0, errors.New("something went wrong")
                }
                return i, nil
            }),
        )
    }),
    Share[int](),
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
// Source execution started...
// Sub1: 1
// Sub2: 1
// Sub1: 2
// Sub2: 2
// Sub1 Error: something went wrong
// Sub2 Error: something went wrong
```

### With hot observable

```go
// Create a hot observable (starts immediately)
hotSource := ro.Pipe[int64, int64](
    ro.Interval(100 * time.Millisecond),
    Take[int64](10),
    ro.Share[int64](), // Make it hot and shareable
)

// Multiple subscribers can join at different times
var subs []*Subscription

for i := 0; i < 3; i++ {
    go func(idx int) {
        time.Sleep(time.Duration(idx) * 150 * time.Millisecond)
        sub := hotSource.Subscribe(ro.OnNext(func(value int64) {
            fmt.Printf("Subscriber %d: %d\n", idx, value)
        }))
        time.Sleep(500 * time.Millisecond)
        sub.Unsubscribe()
    }(i)
}

time.Sleep(1200 * time.Millisecond)

// Each subscriber starts at different times but gets shared values
// from the point they subscribe onward
```

### With reference counting

```go
source := ro.Pipe[int64, int64](
    ro.Interval(100 * time.Millisecond),
    ro.Share[int64](),
)

fmt.Println("Creating first subscription...")
sub1 := source.Subscribe(ro.PrintObserver[int64]())

time.Sleep(250 * time.Millisecond)

fmt.Println("Creating second subscription...")
sub2 := source.Subscribe(ro.PrintObserver[int64]())

time.Sleep(250 * time.Millisecond)

fmt.Println("Unsubscribing first...")
sub1.Unsubscribe()

time.Sleep(250 * time.Millisecond)

fmt.Println("Unsubscribing second...")
sub2.Unsubscribe()

fmt.Println("All subscriptions done")

// The shared observable manages reference counting automatically
// Values are emitted while at least one subscription is active
```