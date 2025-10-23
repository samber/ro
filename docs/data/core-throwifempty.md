---
name: ThrowIfEmpty
slug: throwifempty
sourceRef: operator_error_handling.go#L229
type: core
category: error-handling
signatures:
  - "func ThrowIfro.Empty[T any](throw func() error)"
playUrl:
variantHelpers:
  - core#error-handling#throwifempty
similarHelpers:
  - core#error-handling#onerrorreturn
  - core#conditional#defaultifempty
position: 20
---

Throws an error if the source observable is empty, otherwise emits all items normally.

```go
obs := ro.Pipe[int, int](
    ro.Empty[int](),
    ro.ThrowIfEmpty[int](func() error {
        return errors.New("no data available")
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Error: no data available
```

### With data present

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.ThrowIfEmpty[int](func() error {
        return errors.New("this won't be thrown")
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Completed (no error thrown)
```

### With filtered data

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.Filter(func(i int) bool {
        return i > 10 // No items match
    }),
    ro.ThrowIfEmpty[int](func() error {
        return errors.New("no items found matching criteria")
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Error: no items found matching criteria
```

### With API response validation

```go
type User struct {
    ID   int
    Name string
}

fetchUsers := func() Observable[User] {
    // Simulate empty API response
    return ro.FromSlice([]User{})
}

obs := ro.Pipe[User, User](
    fetchUsers(),
    ro.ThrowIfEmpty[User](func() error {
        return errors.New("no users found in database")
    }),
)

sub := obs.Subscribe(ro.PrintObserver[User]())
defer sub.Unsubscribe()

// Error: no users found in database
```

### With conditional throwing

```go
shouldThrowError := true
obs := ro.Pipe[string, string](
    ro.Empty[string](),
    ro.ThrowIfEmpty[string](func() error {
        if shouldThrowError {
            return fmt.Errorf("empty sequence not allowed at %v", time.Now())
        }
        return nil // No error
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Error: empty sequence not allowed at [current time]
```

### With retry mechanism

```go
attempt := 0
getData := func() Observable[int] {
    attempt++
    if attempt < 3 {
        return ro.Empty[int]() // Simulate empty response
    }
    return ro.Just(42) // Success on third attempt
}

obs := ro.Pipe[int, int](
    ro.Defer(getData),
    ro.ThrowIfEmpty[int](func() error {
        return fmt.Errorf("attempt %d: no data available", attempt)
    }),
    ro.RetryWithConfig[int](RetryConfig{
        MaxRetries: 5,
        Delay:      100 * time.Millisecond,
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
time.Sleep(500 * time.Millisecond)
sub.Unsubscribe()

// Error: attempt 1: no data available
// Error: attempt 2: no data available
// Next: 42 (success on third attempt)
// Completed
```