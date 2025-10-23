---
name: Empty
slug: empty
sourceRef: operator_creation.go#L38
type: core
category: creation
signatures:
  - "func Empty[T any]()"
playUrl:
variantHelpers:
  - core#creation#empty
similarHelpers:
  - core#creation#never
  - core#creation#throw
position: 31
---

Creates an Observable that emits no items and immediately completes. This creation operator is very useful for unit tests.

```go
obs := ro.Empty[int]()

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// No items emitted
// Completed
```

### As a source for other operators

```go
obs := ro.Pipe[int, int](
    ro.Empty[int](),
    ro.DefaultIfEmpty(-1),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: -1 (default value since source is empty)
// Completed
```

### For unit tests

```go
// pipeline.go
var pipeline ro.PipeOp(
    ro.Map(func(x int) int {
        return x*2
    })
    ro.Catch(func(err error) ro.Observable[int] {
        return ro.Just(42)
    }),
)

// pipeline_test.go
func TestMyPipeline(t *testing.T) {
    // testing empty source
    obs := pipeline(ro.Empty[int]())

    values, err := ro.Collect(obs)
    defer sub.Unsubscribe()

    t.Assert(...)

    // testing broken source
    obs := pipeline(ro.Throw[int](errors.New("something went wrong")))

    values, err = ro.Collect(obs)
    defer sub.Unsubscribe()

    t.Assert(...)

    // testing inactive stream
    obs := pipeline(ro.Never[int]())

    values, err = ro.Collect(obs)
    defer sub.Unsubscribe()

    t.Assert(...)
}
```
