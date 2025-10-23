---
name: Defer
slug: defer
sourceRef: operator_creation.go#L422
type: core
category: creation
signatures:
  - "func Defer[T any](factory func() Observable[T])"
playUrl:
variantHelpers:
  - core#creation#defer
similarHelpers:
  - core#creation#future
  - core#creation#start
position: 34
---

Creates an Observable that calls the specified factory function for each subscriber to create a new Observable.

```go
obs := ro.Defer(func() ro.Observable[int] {
    return ro.Just(1, 2, 3)
})

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Completed
```

### Fresh observable for each subscriber

```go
obs := ro.Defer(func() ro.Observable[int] {
    counter := 0
    return ro.Pipe[int64, int](
        ro.Interval(100*time.Millisecond),
        ro.Take[int64](3),
        ro.Map(func(i int64) int {
            counter++
            return counter
        }),
    )
})

// Each subscriber gets a fresh observable
sub1 := obs.Subscribe(ro.PrintObserver[int]())
sub2 := obs.Subscribe(ro.PrintObserver[int]())

time.Sleep(500 * time.Millisecond)
sub1.Unsubscribe()
sub2.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3 (each subscriber gets their own counter)
// Completed
```

### With dynamic content

```go
obs := ro.Defer(func() ro.Observable[string] {
    timestamp := time.Now().Format("15:04:05")
    return ro.Just(fmt.Sprintf("Created at %s", timestamp))
})

sub1 := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(100 * time.Millisecond)
sub2 := obs.Subscribe(ro.PrintObserver[string]())

defer sub1.Unsubscribe()
defer sub2.Unsubscribe()

// Next: "Created at HH:MM:SS" (different times for sub1 and sub2)
```

### For resource cleanup

```go
obs := ro.Defer(func() ro.Observable[string] {
    file, err := os.Open("example.txt")
    if err != nil {
        return ro.Throw[string](err)
    }

    // Return observable that cleans up when done
    return ro.Pipe[string, string](
        ro.FromSlice([]string{"line1", "line2", "line3"}),
        ro.Finally(func() {
            file.Close()
        }),
    )
})

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: line1
// Next: line2
// Next: line3
// Completed (file closed automatically)
```

### Conditional observable creation

```go
obs := ro.Defer(func() ro.Observable[int] {
    if time.Now().Hour() < 12 {
        return ro.Just(1) // Morning
    } else if time.Now().Hour() < 18 {
        return ro.Just(2) // Afternoon
    } else {
        return ro.Just(3) // Evening
    }
})

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1, 2, or 3 depending on current time
// Completed
```