---
name: WindowWhen
slug: windowwhen
sourceRef: operator_transformations.go#L593
type: core
category: transformation
signatures:
  - "func WindowWhen[T any, B any](boundary Observable[B])"
playUrl: https://go.dev/play/p/-FU2r4-mEhz
variantHelpers:
  - core#transformation#windowwhen
similarHelpers: [core#transformation#bufferwhen, core#transformation#windowwithtime, core#transformation#windowwithcount]
position: 85
---

Branches out the source Observable values as a nested Observable whenever the boundary Observable emits an item, and a new window opens when the boundary Observable emits an item.

```go
boundary := ro.Interval(2000*time.Millisecond)

obs := ro.Pipe[int64, ro.Observable[int64]](
    ro.Interval(500*time.Millisecond),
    ro.WindowWhen(boundary),
    ro.Take(3),
)

sub := obs.Subscribe(ro.NewObserver[ro.Observable[int64]](
    func(window ro.Observable[int64]) {
        fmt.Println("New window opened")

        windowSub := window.Subscribe(ro.PrintObserver[int64]())
        time.Sleep(2500 * time.Millisecond)
        windowSub.Unsubscribe()
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
time.Sleep(7000 * time.Millisecond)
sub.Unsubscribe()

// New window opened
// Next: 0, 1, 2, 3 (emitted over 2 seconds)
// New window opened
// Next: 4, 5, 6, 7 (emitted over 2 seconds)
// New window opened
// Next: 8, 9, 10, 11 (emitted over 2 seconds)
// Completed
```

### With time-based boundary

```go
boundary := ro.Timer(1000*time.Millisecond) // Window closes after 1 second

obs := ro.Pipe[string, ro.Observable[string]](
    ro.Just("a", "b", "c", "d", "e", "f"),
    ro.WindowWhen(boundary),
)

sub := obs.Subscribe(ro.NewObserver[ro.Observable[string]](
    func(window ro.Observable[string]) {
        fmt.Println("New window:")
        windowSub := window.Subscribe(ro.PrintObserver[string]())
        windowSub.Unsubscribe()
    },
))
defer sub.Unsubscribe()

// New window:
// Next: a, b, c, d, e, f (all items before boundary)
// Completed
```

### With count-based boundary

```go
boundary := ro.Pipe[int64, string](
    ro.Interval(500*time.Millisecond),
    ro.Map(func(_ int64) string { return "close" }),
)

obs := ro.Pipe[string, ro.Observable[string]](
    ro.Just("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"),
    ro.WindowWhen(boundary),
)

sub := obs.Subscribe(ro.NewObserver[ro.Observable[string]](
    func(window ro.Observable[string]) {
        fmt.Println("Window opened:")
        windowSub := window.Subscribe(ro.PrintObserver[string]())
        time.Sleep(200 * time.Millisecond)
        windowSub.Unsubscribe()
    },
))
defer sub.Unsubscribe()

// Window opened:
// Next: x1 (first item before first boundary)
// Window opened:
// Next: x2 (second item before second boundary)
// Window opened:
// Next: x3 (third item before third boundary)
// ... and so on
```

### With complex boundary conditions

```go
// Boundary emits every 3 source items
counter := 0
boundary := ro.Pipe[string, int](
    ro.Just("trigger"),
    ro.Map(func(_ string) int {
        counter++
        return counter
    }),
    ro.Filter(func(c int) bool { return c%3 == 0 }),
)

obs := ro.Pipe[string, ro.Observable[string]](
    ro.Just("a", "b", "c", "d", "e", "f", "g", "h", "i"),
    ro.WindowWhen(boundary),
)

sub := obs.Subscribe(ro.NewObserver[ro.Observable[string]](
    func(window ro.Observable[string]) {
        fmt.Println("Window:")
        windowSub := window.Subscribe(ro.PrintObserver[string]())
        windowSub.Unsubscribe()
    },
))
defer sub.Unsubscribe()

// Window:
// Next: a, b, c (first 3 items)
// Window:
// Next: d, e, f (next 3 items)
// Window:
// Next: g, h, i (last 3 items)
// Completed
```

### With error in source

```go
boundary := ro.Timer(1000*time.Millisecond)

obs := ro.Pipe[string, ro.Observable[string]](
    ro.Pipe[string, string](
        ro.Just("will error"),
        ro.Throw[string](errors.New("source error")),
    ),
    ro.WindowWhen(boundary),
)

sub := obs.Subscribe(ro.NewObserver[ro.Observable[string]](
    func(window ro.Observable[string]) {
        fmt.Println("Window opened")
        windowSub := window.Subscribe(ro.PrintObserver[string]())
        windowSub.Unsubscribe()
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
))
defer sub.Unsubscribe()

// Error: source error
```