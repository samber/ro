---
name: CombineLatestWith
slug: combinelatestwith
sourceRef: operator_combining.go#L26
type: core
category: combining
signatures:
  - "func CombineLatestWith[A any, B any](obsB Observable[B])"
  - "func CombineLatestWith1[A any, B any](obsB Observable[B])"
  - "func CombineLatestWith2[A any, B any, C any](obsB Observable[B], obsC Observable[C])"
  - "func CombineLatestWith3[A any, B any, C any, D any](obsB Observable[B], obsC Observable[C], obsD Observable[D])"
  - "func CombineLatestWith4[A any, B any, C any, D any, E any](obsB Observable[B], obsC Observable[C], obsD Observable[D], obsE Observable[E])"
playUrl:
variantHelpers:
  - core#combining#combinelatestwith
similarHelpers:
  - core#combining#combinelatest
  - core#combining#combinelatestall
  - core#combining#zipwith
position: 20
---

Creates an Observable that combines the latest values from the source Observable with other provided Observables, emitting tuples of the most recent values.

```go
obs := ro.Pipe[int, lo.Tuple2[int, string]](
    ro.Just(1, 2, 3),
    ro.CombineLatestWith(ro.Just("A", "B", "C")),
)

sub := obs.Subscribe(ro.PrintObserver[lo.Tuple2[int, string]]())
defer sub.Unsubscribe()

// Next: (1, A)
// Next: (2, A)
// Next: (2, B)
// Next: (3, B)
// Next: (3, C)
// Completed
```

### CombineLatestWith2 (three observables)

```go
obs := ro.Pipe[int, lo.Tuple3[int, string, bool]](
    ro.Just(1, 2, 3),
    ro.CombineLatestWith2(
        ro.Just("A", "B", "C"),
        ro.Just(true, false),
    ),
)

sub := obs.Subscribe(ro.PrintObserver[lo.Tuple3[int, string, bool]]())
defer sub.Unsubscribe()

// Next: (1, A, true)
// Next: (2, A, true)
// Next: (2, B, true)
// Next: (3, B, true)
// Next: (3, B, false)
// Next: (3, C, false)
// Completed
```

### Chain multiple CombineLatestWith

```go
obs := ro.Pipe[int, lo.Tuple3[int, string, bool]](
    ro.Just(1, 2, 3),
    ro.CombineLatestWith(ro.Just("A", "B")),
    ro.CombineLatestWith(ro.Just(true, false)),
)

sub := obs.Subscribe(ro.PrintObserver[lo.Tuple3[int, string, bool]]())
defer sub.Unsubscribe()

// Same result as CombineLatestWith2
// Next: (1, A, true)
// Next: (2, A, true)
// Next: (2, B, true)
// Next: (3, B, true)
// Next: (3, B, false)
// Completed
```

### CombineLatestWith3 (four observables)

```go
obs := ro.Pipe[int, lo.Tuple4[int, string, bool, float64]](
    ro.Just(1, 2),
    ro.CombineLatestWith3(
        ro.Just("A", "B"),
        ro.Just(true, false),
        ro.Just(1.1, 2.2),
    ),
)

sub := obs.Subscribe(ro.PrintObserver[lo.Tuple4[int, string, bool, float64]]())
defer sub.Unsubscribe()

// Combines latest from all four sources
// Next: (1, A, true, 1.1)
// Next: (2, A, true, 1.1)
// Next: (2, B, true, 1.1)
// Next: (2, B, false, 1.1)
// Next: (2, B, false, 2.2)
// Completed
```

### Practical example: Combining data streams

```go
// User IDs stream
userIDs := ro.Just(1, 2, 3)

// User details stream (slower)
userDetails := ro.Pipe[User, int](
    ro.Just(
        User{1, "Alice"},
        User{2, "Bob"},
        User{3, "Charlie"},
    ),
    ro.Map(func(u User) int { return u.ID }), // Extract IDs
)

// Permissions stream
permissions := ro.Just("admin", "user", "guest")

// Combine user IDs with their details and permissions
obs := ro.Pipe[int, string](
    userIDs,
    ro.CombineLatestWith2(userDetails, permissions),
    ro.Map(func(t lo.Tuple3[int, int, string]) string {
        userID, detailID, perm := t.Get3()
        return fmt.Sprintf("User %d (detail %d): %s", userID, detailID, perm)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Outputs combinations of latest values from all streams
```