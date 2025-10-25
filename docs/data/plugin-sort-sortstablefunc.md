---
name: SortStableFunc
slug: sortstablefunc
sourceRef: plugins/sort/operator.go#L82
type: plugin
category: sort
signatures:
  - "func SortStableFunc[T comparable](cmp func(a, b T) int)"
playUrl: https://go.dev/play/p/DNXdPAF0TBh
variantHelpers:
  - plugin#sort#sortstablefunc
similarHelpers:
  - plugin#sort#sort
  - plugin#sort#sortfunc
position: 20
---

Sorts elements using a stable sort with custom comparison function. Stable sort preserves the relative order of equal elements.

```go
import (
    "github.com/samber/ro"
    rosort "github.com/samber/ro/plugins/sort"
)

type Person struct {
    Name string
    Age  int
}

people := []Person{
    {"Alice", 25},
    {"Bob", 30},
    {"Charlie", 25},
    {"David", 30},
}

obs := ro.Pipe[Person, Person](
    ro.Just(people...),
    rosort.SortStableFunc(func(a, b Person) int {
        if a.Age != b.Age {
            return a.Age - b.Age // Sort by age
        }
        return 0 // Keep original order for same age
    }),
)

sub := obs.Subscribe(ro.PrintObserver[Person]())
defer sub.Unsubscribe()

// Next: {Alice 25}
// Next: {Charlie 25}  // Same age as Alice, original order preserved
// Next: {Bob 30}
// Next: {David 30}    // Same age as Bob, original order preserved
// Completed
```