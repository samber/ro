---
name: DoWhile
slug: dowhile
sourceRef: operator_error_handling.go#L260
type: core
category: error-handling
signatures:
  - "func DoWhile[T any](condition func() bool)"
  - "func DoWhileI[T any](condition func(index int64) bool)"
  - "func DoWhileWithContext[T any](condition func(context.Context) (context.Context, bool))"
  - "func DoWhileIWithContext[T any](condition func(context.Context, index int64) (context.Context, bool))"
playUrl:
variantHelpers:
  - core#error-handling#dowhile
  - core#error-handling#dowhilei
  - core#error-handling#dowhilewithcontext
  - core#error-handling#dowhileiwithcontext
similarHelpers:
  - core#error-handling#while
position: 50
---

Emits values from the source observable, then repeats the sequence as long as the condition returns true.

```go
counter := 0
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.DoWhile(func() bool {
        counter++
        return counter <= 3
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1, 2, 3 (1st iteration)
// Next: 1, 2, 3 (2nd iteration)
// Next: 1, 2, 3 (3rd iteration)
// Completed
```

### DoWhileI with index

```go
obs := ro.Pipe[string, string](
    ro.Just("a", "b"),
    ro.DoWhileI(func(index int64) bool {
        return index < 2 // Repeat twice (index 0 and 1)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: "a", "b" (index 0)
// Next: "a", "b" (index 1)
// Completed
```

### DoWhileWithContext with cancellation

```go
ctx, cancel := context.WithCancel(context.Background())

obs := ro.Pipe[int, int](
    ro.Just(1, 2),
    ro.DoWhileWithContext(func(ctx context.Context) (context.Context, bool) {
        select {
        case <-ctx.Done():
            return ctx, false
        default:
            return ctx, true // Continue repeating
        }
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())

// After some items...
cancel() // Stop repeating
defer sub.Unsubscribe()
```

### DoWhileIWithContext with index and context

```go
ctx := context.Background()
obs := ro.Pipe[string, string](
    ro.Just("x"),
    ro.DoWhileIWithContext(func(ctx context.Context, index int64) (context.Context, bool) {
        fmt.Printf("Iteration %d\n", index)
        return ctx, index < 2 // Repeat for 2 iterations
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Iteration 0
// Next: "x"
// Iteration 1
// Next: "x"
// Completed
```

### Retry pattern with DoWhile

```go
maxAttempts := 3
attempt := 0
shouldRetry := func() bool {
    attempt++
    return attempt <= maxAttempts
}

obs := ro.Pipe[int, int](
    ro.Defer(func() ro.Observable[int] {
        if attempt < maxAttempts {
            return ro.Throw[int](errors.New("temporary failure"))
        }
        return ro.Just(42) // Success on final attempt
    }),
    ro.DoWhile(shouldRetry),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Would retry until maxAttempts reached
// Next: 42 (success on 3rd attempt)
```

### Polling with DoWhile

```go
ticker := time.NewTicker(100 * time.Millisecond)
defer ticker.Stop()

obs := ro.Pipe[int, int](
    ro.Defer(func() ro.Observable[int] {
        // Simulate checking for new data
        if rand.Intn(10) == 0 {
            return ro.Just(rand.Intn(100))
        }
        return ro.Empty[int]()
    }),
    ro.DoWhileWithContext(func(ctx context.Context) (context.Context, bool) {
        select {
        case <-ticker.C:
            return ctx, true // Continue polling
        case <-ctx.Done():
            return ctx, false
        }
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
time.Sleep(1 * time.Second)
sub.Unsubscribe()

// Emits random values as they become available
// Stops after 1 second
```

### With external state

```go
type GameState struct {
    Score    int
    Lives    int
    GameOver bool
}

game := &GameState{Lives: 3}
obs := ro.Pipe[string, string](
    ro.Defer(func() ro.Observable[string] {
        if game.Lives <= 0 {
            game.GameOver = true
            return ro.Just("Game Over")
        }
        action := fmt.Sprintf("Action - Lives: %d", game.Lives)
        game.Lives--
        return ro.Just(action)
    }),
    ro.DoWhile(func() bool {
        return !game.GameOver
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: "Action - Lives: 3"
// Next: "Action - Lives: 2"
// Next: "Action - Lives: 1"
// Next: "Game Over"
// Completed
```