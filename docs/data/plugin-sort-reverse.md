---
name: Reverse
slug: reverse
sourceRef: plugins/sort/operator.go#L126
type: plugin
category: sort
signatures:
  - "func Reverse[T any]()"
playUrl:
variantHelpers:
  - plugin#sort#reverse
similarHelpers:
  - plugin#sort#sort
  - core#filtering#takelast
  - core#sink#toslice
position: 30
---

Buffers every item emitted by the source until it completes, then re-emits them one by one in reverse order.

```go
import (
    "github.com/samber/ro"
    rosort "github.com/samber/ro/plugins/sort"
)

obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    rosort.Reverse[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 5
// Next: 4
// Next: 3
// Next: 2
// Next: 1
// Completed
```

### Memory usage note

```go
// Reverse must buffer all elements before emitting the first one.
// Never use it on an unbounded or very large source.
obs := ro.Pipe[string, string](
    ro.Just("apple", "banana", "cherry"),
    rosort.Reverse[string](),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: cherry
// Next: banana
// Next: apple
// Completed
```
