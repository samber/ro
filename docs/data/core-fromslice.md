---
name: FromSlice
slug: fromslice
sourceRef: operator_creation.go#L114
type: core
category: creation
signatures:
  - "func FromSlice[T any](slice []T)"
playUrl:
variantHelpers:
  - core#creation#fromslice
similarHelpers:
  - core#creation#fromchannel
  - core#creation#just
position: 40
---

Creates an Observable that emits each item from a slice, then completes.

```go
data := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
obs := ro.Pipe[int, int](
    ro.FromSlice(data),
    ro.Filter(func(n int) bool { return n%2 == 0 }),
    ro.Map(func(n int) int { return n * n }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 4
// Next: 16
// Next: 36
// Next: 64
// Next: 100
// Completed
```

### With complex types

```go
type Person struct {
    Name string
    Age  int
}

people := []Person{
    {"Alice", 25},
    {"Bob", 30},
    {"Charlie", 35},
}

obs := ro.FromSlice(people)

sub := obs.Subscribe(ro.PrintObserver[Person]())
defer sub.Unsubscribe()

// Next: {Alice 25}
// Next: {Bob 30}
// Next: {Charlie 35}
// Completed
```

### Fresh slice for each subscriber

```go
obs := ro.Defer(func() ro.Observable[int] {
    return ro.FromSlice([]int{
        rand.Intn(100),
        rand.Intn(100),
    })
})

sub1 := obs.Subscribe(ro.PrintObserver[int]())
sub2 := obs.Subscribe(ro.PrintObserver[int]())

defer sub1.Unsubscribe()
defer sub2.Unsubscribe()

// Each subscriber gets potentially different random numbers
```
