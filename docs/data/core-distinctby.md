---
name: DistinctBy
slug: distinctby
sourceRef: operator_filter.go#L98
type: core
category: filtering
signatures:
  - "func DistinctBy[T any, K comparable](keySelector func(item T) K)"
  - "func DistinctByWithContext[T any, K comparable](keySelector func(ctx context.Context, item T) (context.Context, K))"
playUrl:
variantHelpers:
  - core#filtering#distinctby
  - core#filtering#distinctbywithcontext
similarHelpers:
  - core#filtering#distinct
position: 62
---

Suppresses duplicate items in an Observable based on a key selector function.

```go
type user struct {
    id   int
    name string
}

obs := ro.Pipe(
    ro.Just(
        user{id: 1, name: "John"},
        user{id: 2, name: "Jane"},
        user{id: 1, name: "John"},
        user{id: 3, name: "Jim"},
    ),
    ro.DistinctBy(func(item user) int {
        return item.id
    }),
)

sub := obs.Subscribe(ro.PrintObserver[user]())
defer sub.Unsubscribe()

// Next: {1 John}
// Next: {2 Jane}
// Next: {3 Jim}
// Completed
```

## With string key selector

```go
obs := ro.Pipe(
    ro.Just("apple", "banana", "apple", "cherry", "banana"),
    ro.DistinctBy(func(item string) string {
        return item
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: apple
// Next: banana
// Next: cherry
// Completed
```

## With complex key selector

```go
type product struct {
    category string
    name     string
    price    float64
}

obs := ro.Pipe(
    ro.Just(
        product{category: "electronics", name: "laptop", price: 999.99},
        product{category: "clothing", name: "shirt", price: 29.99},
        product{category: "electronics", name: "phone", price: 699.99},
        product{category: "electronics", name: "laptop", price: 1099.99},
    ),
    ro.DistinctBy(func(item product) string {
        return item.category + ":" + item.name
    }),
)

sub := obs.Subscribe(ro.PrintObserver[product]())
defer sub.Unsubscribe()

// Next: {electronics laptop 999.99}
// Next: {clothing shirt 29.99}
// Next: {electronics phone 699.99}
// Completed
```

## With context-aware key selector

```go
import (
    "context"
    "fmt"
    "time"
)

type event struct {
    id        int
    timestamp time.Time
    data      string
}

obs := ro.Pipe(
    ro.Just(
        event{id: 1, timestamp: time.Now(), data: "event1"},
        event{id: 2, timestamp: time.Now().Add(time.Hour), data: "event2"},
        event{id: 1, timestamp: time.Now().Add(2*time.Hour), data: "event1_duplicate"},
        event{id: 3, timestamp: time.Now().Add(3*time.Hour), data: "event3"},
    ),
    ro.DistinctByWithContext(func(ctx context.Context, item event) (context.Context, int) {
        // Context can be used for logging, tracing, or other cross-cutting concerns
        fmt.Printf("Processing event %d with context: %v\n", item.id, ctx.Value("requestId"))
        return ctx, item.id
    }),
)

sub := obs.Subscribe(ro.PrintObserver[event]())
defer sub.Unsubscribe()

// Processing event 1 with context: <nil>
// Processing event 2 with context: <nil>
// Processing event 1 with context: <nil>
// Processing event 3 with context: <nil>
// Next: {1 2024-01-01T10:00:00Z event1}
// Next: {2 2024-01-01T11:00:00Z event2}
// Next: {3 2024-01-01T13:00:00Z event3}
// Completed
```

## With context propagation

```go
import (
    "context"
    "fmt"
)

type task struct {
    id     string
    status string
    user   string
}

// Create a context with a user ID
ctx := context.WithValue(context.Background(), "userId", "user123")

obs := ro.Pipe(
    ro.Just(
        task{id: "task1", status: "pending", user: "user123"},
        task{id: "task2", status: "completed", user: "user456"},
        task{id: "task1", status: "in_progress", user: "user123"},
        task{id: "task3", status: "pending", user: "user789"},
    ),
    ro.DistinctByWithContext(func(ctx context.Context, item task) (context.Context, string) {
        // Use context to filter by user or add user-specific logic
        userId := ctx.Value("userId").(string)
        if item.user == userId {
            fmt.Printf("Processing task %s for user %s\n", item.id, userId)
        }
        return ctx, item.id
    }),
)

sub := obs.Subscribe(ro.PrintObserver[task]())
defer sub.Unsubscribe()

// Processing task task1 for user user123
// Processing task task2 for user user123
// Processing task task1 for user user123
// Processing task task3 for user user123
// Next: {task1 pending user123}
// Next: {task2 completed user456}
// Next: {task3 pending user789}
// Completed
```
