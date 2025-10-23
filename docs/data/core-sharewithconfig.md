---
name: ShareWithConfig
slug: sharewithconfig
sourceRef: operator_connectable.go#L64
type: core
category: connectable
signatures:
  - "func ShareWithConfig[T any](config ShareConfig[T])"
playUrl:
variantHelpers:
  - core#connectable#sharewithconfig
similarHelpers:
  - core#connectable#share
  - core#connectable#sharereplay
position: 10
---

Creates a shared Observable with customizable configuration. Allows fine-grained control over subject selection and reset behavior.

```go
config := ShareConfig[string]{
    Connector:           func() Subject[string] { return ro.NewPublishSubject[string]() },
    ResetOnError:        true,
    ResetOnComplete:     true,
    ResetOnRefCountZero: true,
}

obs := ro.Pipe[string, string](
    ro.Just("hello", "world"),
    ro.ShareWithConfig(config),
)

sub1 := obs.Subscribe(ro.PrintObserver[string]())
sub2 := obs.Subscribe(ro.PrintObserver[string]())

time.Sleep(100 * time.Millisecond)
sub1.Unsubscribe()
sub2.Unsubscribe()
```

### With custom subject

```go
// Custom subject that logs all operations
type LoggingSubject struct {
    *PublishSubject[string]
    id string
}

func NewLoggingSubject(id string) *LoggingSubject {
    return &LoggingSubject{
        PublishSubject: ro.NewPublishSubject[string](),
        id:           id,
    }
}

config := ShareConfig[string]{
    Connector: func() Subject[string] {
        subject := ro.NewLoggingSubject("custom-subject")
        fmt.Printf("Created new logging subject: %s\n", subject.id)
        return subject
    },
    ResetOnError: true,
}

obs := ro.Pipe[string, string](
    ro.Just("data1", "data2"),
    ro.ShareWithConfig(config),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(50 * time.Millisecond)
sub.Unsubscribe()
```

### With error reset behavior

```go
config := ShareConfig[int]{
    Connector:       func() Subject[int] { return ro.NewPublishSubject[int]() },
    ResetOnError:    true, // Reset on error
    ResetOnComplete: false,
}

source := ro.Pipe[int, int](
    ro.Defer(func() Observable[int] {
        fmt.Println("Starting new source execution...")
        return ro.Pipe[int, int](
            ro.Just(1, 2, 3),
            ro.MapErr(func(i int) (int, error) {
                if i == 3 {
                    return 0, errors.New("test error")
                }
                return i, nil
            }),
        )
    }),
    ro.ShareWithConfig(config),
)

// First subscriber - will trigger error
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

// Second subscriber - gets fresh source due to ResetOnError
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
// Starting new source execution...
// Sub1: 1
// Sub1: 2
// Sub1 Error: test error
// Starting new source execution... (ResetOnError triggered)
// Sub2: 1
// Sub2: 2
// Sub2 Error: test error
```

### With completion reset behavior

```go
config := ShareConfig[string]{
    Connector:       func() Subject[string] { return ro.NewPublishSubject[string]() },
    ResetOnError:    false,
    ResetOnComplete: true, // Reset on completion
}

source := ro.Pipe[string, string](
    ro.Defer(func() Observable[string] {
        fmt.Println("New source execution...")
        return ro.Just("once", "twice")
    }),
    ro.ShareWithConfig(config),
)

sub1 := source.Subscribe(ro.PrintObserver[string]())
time.Sleep(50 * time.Millisecond)
sub1.Unsubscribe()

// Second subscriber gets fresh source due to ResetOnComplete
sub2 := source.Subscribe(ro.PrintObserver[string]())
time.Sleep(50 * time.Millisecond)
sub2.Unsubscribe()

// 
// New source execution...
// Next: once, Next: twice
// New source execution... (ResetOnComplete triggered)
// Next: once, Next: twice
```

### With ref count zero behavior

```go
config := ShareConfig[int]{
    Connector:           func() Subject[int] { return ro.NewPublishSubject[int]() },
    ResetOnError:        false,
    ResetOnComplete:     false,
    ResetOnRefCountZero: true, // Reset when no subscribers left
}

source := ro.Pipe[int64, int64](
    ro.Defer(func() Observable[int64] {
        fmt.Println("Source created...")
        return ro.Interval(100 * time.Millisecond).ro.Take[int64](5)
    }),
    ro.ShareWithConfig(config),
)

fmt.Println("First subscriber...")
sub1 := source.Subscribe(ro.OnNext(func(value int64) {
    fmt.Printf("Sub1: %d\n", value)
}))

time.Sleep(350 * time.Millisecond)
fmt.Println("Unsubscribing first...")
sub1.Unsubscribe()

time.Sleep(100 * time.Millisecond)

fmt.Println("Second subscriber...")
sub2 := source.Subscribe(ro.OnNext(func(value int64) {
    fmt.Printf("Sub2: %d\n", value)
}))

time.Sleep(350 * time.Millisecond)
fmt.Println("Unsubscribing second...")
sub2.Unsubscribe()

// 
// First subscriber...
// Source created...
// Sub1: 0, Sub1: 1, Sub1: 2
// Unsubscribing first...
// Second subscriber...
// Source created... (ResetOnRefCountZero triggered)
// Sub2: 0, Sub2: 1, Sub2: 2
// Unsubscribing second...
```

### With BehaviorSubject for initial value

```go
config := ShareConfig[int]{
    Connector: func() Subject[int] {
        // Use BehaviorSubject with initial value
        return ro.NewBehaviorSubject[int](42)
    },
    ResetOnError:    false,
    ResetOnComplete: false,
}

obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.ShareWithConfig(config),
)

// First subscriber gets immediate initial value
sub1 := obs.Subscribe(ro.OnNext(func(value int) {
    fmt.Printf("Sub1: %d\n", value)
}))

time.Sleep(50 * time.Millisecond)

// Second subscriber gets current value
sub2 := obs.Subscribe(ro.OnNext(func(value int) {
    fmt.Printf("Sub2: %d\n", value)
}))

time.Sleep(50 * time.Millisecond)
sub1.Unsubscribe()
sub2.Unsubscribe()

// 
// Sub1: 42 (initial value from BehaviorSubject)
// Sub1: 1, Sub1: 2, Sub1: 3
// Sub2: 3 (current value)
```

### With ReplaySubject for caching

```go
config := ShareConfig[string]{
    Connector: func() Subject[string] {
        // Use ReplaySubject to cache last 2 values
        return ro.NewReplaySubject[string](2)
    },
    ResetOnError:    false,
    ResetOnComplete: false,
}

obs := ro.Pipe[string, string](
    ro.Just("first", "second", "third"),
    ro.ShareWithConfig(config),
)

sub1 := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(50 * time.Millisecond)
sub1.Unsubscribe()

// Second subscriber gets replayed values
sub2 := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(50 * time.Millisecond)
sub2.Unsubscribe()

// 
// First subscriber: first, second, third
// Second subscriber: second, third (replayed from cache)
```