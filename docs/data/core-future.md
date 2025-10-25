---
name: Future
slug: future
sourceRef: operator_creation.go#L62
type: core
category: creation
signatures:
  - "func Future[T any](promise func(resolve func(T), reject func(error)))"
playUrl: https://go.dev/play/p/BYPTAIeLqFm
variantHelpers:
  - core#creation#future
similarHelpers:
  - core#creation#defer
  - core#creation#start
position: 35
---

Creates an Observable from a promise-style function that can resolve with a value or reject with an error.

```go
obs := ro.Future[string](func(resolve func(string), reject func(error)) {
    go func() {
        time.Sleep(100 * time.Millisecond)
        resolve("Hello from future!")
    }()
})

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(200 * time.Millisecond)
sub.Unsubscribe()

// Next: "Hello from future!" (after 100ms)
// Completed
```

### With error rejection

```go
obs := ro.Future[int](func(resolve func(int), reject func(error)) {
    go func() {
        time.Sleep(50 * time.Millisecond)
        reject(errors.New("something went wrong"))
    }()
})

sub := obs.Subscribe(ro.PrintObserver[int]())
time.Sleep(100 * time.Millisecond)
sub.Unsubscribe()

// Error: something went wrong
```

### HTTP request simulation

```go
obs := ro.Future[string](func(resolve func(string), reject func(error)) {
    go func() {
        // Simulate HTTP request
        time.Sleep(200 * time.Millisecond)

        // Simulate response
        resolve("{\"status\": \"ok\", \"data\": [1, 2, 3]}")
    }()
})

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(300 * time.Millisecond)
sub.Unsubscribe()

// Next: "{\"status\": \"ok\", \"data\": [1, 2, 3]}"
// Completed
```

### With conditional resolution

```go
obs := ro.Future[int](func(resolve func(int), reject func(error)) {
    go func() {
        // Simulate random success/failure
        time.Sleep(100 * time.Millisecond)
        if rand.Intn(2) == 0 {
            resolve(42)
        } else {
            reject(errors.New("random failure"))
        }
    }()
})

sub := obs.Subscribe(ro.PrintObserver[int]())
time.Sleep(200 * time.Millisecond)
sub.Unsubscribe()

// Either:
// Next: 42
// Completed
// Or:
// Error: random failure
```

### Database query simulation

```go
obs := ro.Future[[]string](func(resolve func([]string), reject func(error)) {
    go func() {
        // Simulate database query
        time.Sleep(150 * time.Millisecond)

        // Simulate query results
        resolve([]string{"Alice", "Bob", "Charlie"})
    }()
})

sub := obs.Subscribe(ro.PrintObserver[[]string]())
time.Sleep(300 * time.Millisecond)
sub.Unsubscribe()

// Next: ["Alice", "Bob", "Charlie"]
// Completed
```