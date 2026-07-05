---
name: Collect
slug: collect
sourceRef: observable.go#L327
type: core
category: sink
signatures:
  - "func Collect[T any](obs Observable[T]) ([]T, error)"
  - "func CollectWithContext[T any](ctx context.Context, obs Observable[T]) ([]T, context.Context, error)"
playUrl: https://go.dev/play/p/3rObncYBG2h
variantHelpers:
  - core#sink#collect
  - core#sink#collectwithcontext
similarHelpers:
  - core#sink#toslice
  - core#sink#tomap
position: 0
---

Subscribes to an Observable and collects all emitted values into a slice, blocking until the Observable completes or errors.

This is a blocking sink function — unlike ToSlice, it does not return an Observable and is typically used in tests or when all values need to be gathered before proceeding.

```go
values, err := ro.Collect(ro.Just(1, 2, 3, 4, 5))

fmt.Println(values) // [1 2 3 4 5]
fmt.Println(err)    // <nil>
```

### With context

```go
ctx := context.Background()
values, ctx, err := ro.CollectWithContext(ctx, ro.Just("a", "b", "c"))

fmt.Println(values) // [a b c]
fmt.Println(err)    // <nil>
```
