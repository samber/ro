---
name: DistinctBy
slug: distinctby
sourceRef: operator_filter.go#L98
type: core
category: filtering
signatures:
  - "func DistinctBy[T any, K comparable](keySelector func(item T) K)"
playUrl:
variantHelpers:
  - core#filtering#distinctby
similarHelpers:
  - core#filtering#distinct
position: 60
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
