---
name: GroupBy
slug: groupby
sourceRef: operator_transformations.go#L293
type: core
category: transformation
signatures:
  - "func GroupBy[T any, K comparable](keySelector func(item T) K)"
  - "func GroupByWithContext[T any, K comparable](keySelector func(ctx context.Context, item T) K)"
  - "func GroupByI[T any, K comparable](keySelector func(item T, index int64) K)"
  - "func GroupByIWithContext[T any, K comparable](keySelector func(ctx context.Context, item T, index int64) K)"
playUrl:
variantHelpers:
  - core#transformation#groupby
  - core#transformation#groupbywithcontext
  - core#transformation#groupbyi
  - core#transformation#groupbyiwithcontext
similarHelpers: []
position: 200
---

Groups the items emitted by an observable sequence according to a specified key selector function.

```go
obs := ro.Pipe[string, ro.Observable[string]](
    ro.Just("apple", "banana", "avocado", "blueberry", "cherry"),
    ro.GroupBy(func(fruit string) string {
        return string(fruit[0]) // Group by first letter
    }),
)

sub := obs.Subscribe(ro.PrintObserver[ro.Observable[string]]())
defer sub.Unsubscribe()

// Each emission is an observable of grouped items
// Need to subscribe to each group observable
```

### With context

```go
obs := ro.Pipe[int, ro.Observable[int]](
    ro.Just(1, 2, 3, 4, 5, 6),
    ro.GroupByWithContext(func(ctx context.Context, n int) string {
        if n%2 == 0 {
            return "even"
        }
        return "odd"
    }),
)

sub := obs.Subscribe(ro.PrintObserver[ro.Observable[int]]())
defer sub.Unsubscribe()
```

### With index

```go
obs := ro.Pipe[string, ro.Observable[string]](
    ro.Just("a", "b", "c", "d", "e", "f"),
    ro.GroupByI(func(item string, index int64) int {
        return int(index / 2) // Group by pairs
    }),
)

sub := obs.Subscribe(ro.PrintObserver[ro.Observable[string]]())
defer sub.Unsubscribe()
```

### With index and context

```go
obs := ro.Pipe[string, ro.Observable[string]](
    ro.Just("file1.txt", "image1.jpg", "file2.txt", "image2.jpg"),
    ro.GroupByIWithContext(func(ctx context.Context, filename string, index int64) string {
        if strings.HasSuffix(filename, ".jpg") {
            return "images"
        }
        return "documents"
    }),
)

sub := obs.Subscribe(ro.PrintObserver[ro.Observable[string]]())
defer sub.Unsubscribe()
```

### Processing groups example

```go
// Source observable
source := ro.Just("apple", "apricot", "banana", "blueberry", "cherry")

// Group by first letter
groupedObs := ro.Pipe[string, ro.Observable[string]](
    source,
    ro.GroupBy(func(fruit string) string {
        return string(fruit[0])
    }),
)

// Subscribe to groups
groupedSub := groupedObs.Subscribe(ro.NewObserver[ro.Observable[string]](
    func(group ro.Observable[string]) {
        // Subscribe to each group
        groupSub := group.Subscribe(ro.NewObserver[string](
            func(item string) {
                fmt.Printf("Group item: %s\n", item)
            },
            func(err error) {
                fmt.Printf("Group error: %v\n", err)
            },
            func() {
                fmt.Println("Group completed")
            },
        ))
        defer groupSub.Unsubscribe()
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("All groups completed")
    },
))
defer groupedSub.Unsubscribe()
```