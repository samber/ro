---
name: FromChannel
slug: fromchannel
sourceRef: operator_creation.go#L110
type: core
category: creation
signatures:
  - "func FromChannel[T any](ch <-chan T)"
playUrl:
variantHelpers:
  - core#creation#fromchannel
similarHelpers:
  - core#creation#fromslice
position: 39
---

Creates an Observable that emits items from a channel until the channel is closed.

```go
ch := make(chan string)
go func() {
    ch <- "hello"
    ch <- "world"
    close(ch)
}()

obs := ro.FromChannel(ch)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: "hello"
// Next: "world"
// Completed
```

### With buffered channel

```go
ch := make(chan int, 3)
ch <- 1
ch <- 2
ch <- 3
close(ch)

obs := ro.FromChannel(ch)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Completed
```

### With concurrent writes

```go
ch := make(chan int)
go func() {
    for i := 1; i <= 5; i++ {
        ch <- i
        time.Sleep(100 * time.Millisecond)
    }
    close(ch)
}()

obs := ro.FromChannel(ch)

sub := obs.Subscribe(ro.PrintObserver[int]())
time.Sleep(800 * time.Millisecond)
sub.Unsubscribe()

// Next: 1 (after ~100ms)
// Next: 2 (after ~200ms)
// Next: 3 (after ~300ms)
// Next: 4 (after ~400ms)
// Next: 5 (after ~500ms)
// Completed
```

### With multiple subscribers

```go
ch := make(chan string)
go func() {
    ch <- "shared"
    ch <- "data"
    close(ch)
}()

obs := ro.FromChannel(ch)

sub1 := obs.Subscribe(ro.PrintObserver[string]())
sub2 := obs.Subscribe(ro.PrintObserver[string]())

defer sub1.Unsubscribe()
defer sub2.Unsubscribe()

// Both subscribers compete for the same channel values
// One might get both values, the other gets none
```

### With error channel pattern

```go
type Result struct {
    Value string
    Err   error
}

ch := make(chan Result)
go func() {
    ch <- Result{Value: "success"}
    ch <- Result{Err: errors.New("failed")}
    close(ch)
}()

obs := ro.Pipe[Result, string](
    ro.FromChannel(ch),
    ro.MapErr(func(r Result) (string, error) {
        if r.Err != nil {
            return "", r.Err
        }
        return r.Value, nil
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: "success"
// Error: failed
```