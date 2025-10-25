---
name: MapErr
slug: maperr
sourceRef: operator_transformations.go#L96
type: core
category: transformation
signatures:
  - "func MapErr[T any, R any](project func(item T) (R, error))"
  - "func MapErrWithContext[T any, R any](project func(ctx context.Context, item T) (R, error))"
  - "func MapErrI[T any, R any](project func(item T, index int64) (R, error))"
  - "func MapErrIWithContext[T any, R any](project func(ctx context.Context, item T, index int64) (R, error))"
playUrl: https://go.dev/play/p/x7-KC-SDXr1
variantHelpers:
  - core#transformation#maperr
  - core#transformation#maperrwithcontext
  - core#transformation#maperri
  - core#transformation#maperriwithcontext
similarHelpers:
  - core#transformation#map
position: 100
---

Transforms items emitted by an observable sequence with a function that can return errors.

```go
obs := ro.Pipe[string, int](
    ro.Just("1", "2", "three", "4"),
    ro.MapErr(func(s string) (int, error) {
        return strconv.Atoi(s)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Error: strconv.Atoi: parsing "three": invalid syntax
```

### With context

```go
obs := ro.Pipe[string, string](
    ro.Just("file1.txt", "file2.txt", "invalid.txt"),
    ro.MapErrWithContext(func(ctx context.Context, filename string) (string, error) {
        if !strings.HasSuffix(filename, ".txt") {
            return "", fmt.Errorf("invalid file extension: %s", filename)
        }
        return fmt.Sprintf("processed: %s", filename), nil
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: processed: file1.txt
// Next: processed: file2.txt
// Error: invalid file extension: invalid.txt
```

### With index

```go
obs := ro.Pipe[string, string](
    ro.Just("apple", "banana", "cherry"),
    ro.MapErrI(func(fruit string, index int64) (string, error) {
        if index == 1 {
            return "", fmt.Errorf("skipping item at index %d", index)
        }
        return strings.ToUpper(fruit), nil
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: APPLE
// Error: skipping item at index 1
```

### With index and context

```go
obs := ro.Pipe[string, int](
    ro.Just("test1", "test2", "test3"),
    ro.MapErrIWithContext(func(ctx context.Context, item string, index int64) (int, error) {
        if index > 1 {
            return 0, fmt.Errorf("index %d out of range", index)
        }
        return len(item), nil
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 5
// Next: 5
// Error: index 2 out of range
```