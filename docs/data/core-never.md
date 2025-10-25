---
name: Never
slug: never
sourceRef: operator_creation.go#L42
type: core
category: creation
signatures:
  - "func Never[T any]()"
playUrl: https://go.dev/play/p/GHzcVYaEvN8
variantHelpers:
  - core#creation#never
similarHelpers:
  - core#creation#empty
  - core#creation#throw
position: 32
---

Creates an Observable that never emits any items and never completes. This creation operator is very useful for unit tests.

```go
obs := ro.Never[int]()

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// No items ever emitted
// Never completes
```

### With timeout for testing

```go
obs := ro.Never[string]()

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(100 * time.Millisecond)
sub.Unsubscribe()

// No items emitted during sleep
// Never would have completed if we waited forever
```

Context timeout:

```go
obs := ro.Never[string]()

ctx := context.WithTimeout(100 * time.Millisecond)
sub := obs.SubscribeWithContext(ctx, ro.PrintObserver[string]())
defer sub.Unsubscribe()

// No items emitted during sleep
// Never would have completed if we waited forever
```

### For long-running operations

```go
// Simulate a long-running operation that may never complete
obs := ro.Never[bool]()

sub := obs.Subscribe(ro.PrintObserver[bool]())
// In a real app, you might want to add a timeout
// time.Sleep(5 * time.Second)
// sub.Unsubscribe()
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
