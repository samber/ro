---
name: Materialize
slug: materialize
sourceRef: operator_utility.go#L462
type: core
category: utility
signatures:
  - "func Materialize[T any]()"
playUrl:
variantHelpers:
  - core#utility#materialize
similarHelpers:
  - core#utility#dematerialize
position: 460
---

Converts an observable sequence into an observable of notifications representing the original sequence's events.

```go
obs := ro.Pipe[int, ro.Notification[int]](
    ro.Just(1, 2, 3),
    ro.Materialize[int](),
)

sub := obs.Subscribe(ro.PrintObserver[ro.Notification[int]]())
defer sub.Unsubscribe()

// Next: {Value: 1, HasValue: true}
// Next: {Value: 2, HasValue: true}
// Next: {Value: 3, HasValue: true}
// Next: {Error: nil, HasValue: false}
// Completed
```

### With errors

```go
obs := ro.Pipe[int, ro.Notification[int]](
    ro.Concat(
        ro.Just(1, 2),
        ro.Throw[int](errors.New("test error")),
    ),
    ro.Materialize[int](),
)

sub := obs.Subscribe(ro.PrintObserver[ro.Notification[int]]())
defer sub.Unsubscribe()

// Next: {Value: 1, HasValue: true}
// Next: {Value: 2, HasValue: true}
// Next: {Error: test error, HasValue: false}
// Completed
```

### Processing notifications

```go
obs := ro.Pipe[string, ro.Notification[string]](
    ro.Just("hello", "world"),
    ro.Materialize[string](),
)

sub := obs.Subscribe(ro.NewObserver[ro.Notification[string]](
    func(notification ro.Notification[string]) {
        if notification.HasValue {
            fmt.Printf("Value: %v\n", notification.Value)
        } else if notification.Error != nil {
            fmt.Printf("Error: %v\n", notification.Error)
        } else {
            fmt.Println("Completed")
        }
    },
    func(err error) {
        fmt.Printf("Observer error: %v\n", err)
    },
    func() {
        fmt.Println("Observer completed")
    },
))
defer sub.Unsubscribe()

// Value: hello
// Value: world
// Completed
// Observer completed
```