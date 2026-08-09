# Sort Plugin

The sort plugin provides operators for sorting and reordering values in reactive streams. **⚠️ Warning: These operators load all values into memory before emitting, which is not recommended for large datasets.**

## Installation

```bash
go get github.com/samber/ro/plugins/sort
```

## Operators

### Sort

Sorts values using the default comparison function for ordered types.

```go
import (
    "github.com/samber/ro"
    rosort "github.com/samber/ro/plugins/sort"
)

observable := ro.Pipe1(
    ro.Just(3, 1, 4, 1, 5, 9, 2, 6),
    rosort.Sort[int](),
)

subscription := observable.Subscribe(ro.NewObserver(
    func(value int) {
        fmt.Printf("%d ", value)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("\nCompleted")
    },
))

// Output:
// 1 1 2 3 4 5 6 9
// Completed
```

### SortFunc

Sorts values using a custom comparison function.

```go
type User struct {
    Name string
    Age  int
}

observable := ro.Pipe1(
    ro.Just(
        User{Name: "Alice", Age: 30},
        User{Name: "Bob", Age: 25},
        User{Name: "Charlie", Age: 35},
    ),
    rosort.SortFunc(func(a, b User) int {
        return a.Age - b.Age // Sort by age
    }),
)

subscription := observable.Subscribe(ro.NewObserver(
    func(user User) {
        fmt.Printf("%s (age %d)\n", user.Name, user.Age)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))

// Output:
// Bob (age 25)
// Alice (age 30)
// Charlie (age 35)
// Completed
```

### SortStableFunc

Sorts values using a custom comparison function with stable sorting (preserves relative order of equal elements).

```go
type Event struct {
    Timestamp time.Time
    Priority  int
    Message   string
}

observable := ro.Pipe1(
    ro.Just(
        Event{Timestamp: time.Now(), Priority: 1, Message: "First"},
        Event{Timestamp: time.Now().Add(time.Second), Priority: 2, Message: "Second"},
        Event{Timestamp: time.Now().Add(2 * time.Second), Priority: 1, Message: "Third"},
    ),
    rosort.SortStableFunc(func(a, b Event) int {
        return a.Priority - b.Priority // Sort by priority, stable
    }),
)

subscription := observable.Subscribe(ro.NewObserver(
    func(event Event) {
        fmt.Printf("Priority %d: %s\n", event.Priority, event.Message)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))

// Output:
// Priority 1: First
// Priority 1: Third
// Priority 2: Second
// Completed
```

### Reverse

Buffers every item until the source completes, then re-emits them one by one in reverse order.

```go
observable := ro.Pipe1(
    ro.Just(1, 2, 3, 4, 5),
    rosort.Reverse[int](),
)

subscription := observable.Subscribe(ro.NewObserver(
    func(value int) {
        fmt.Printf("%d ", value)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("\nCompleted")
    },
))

// Output:
// 5 4 3 2 1
// Completed
```

## Advanced Usage

### Sorting with Complex Logic

```go
type Product struct {
    Name     string
    Price    float64
    Category string
}

observable := ro.Pipe1(
    ro.Just(
        Product{Name: "Laptop", Price: 999.99, Category: "Electronics"},
        Product{Name: "Book", Price: 19.99, Category: "Books"},
        Product{Name: "Phone", Price: 699.99, Category: "Electronics"},
        Product{Name: "Pen", Price: 2.99, Category: "Office"},
    ),
    rosort.SortFunc(func(a, b Product) int {
        // Sort by category first, then by price
        if a.Category != b.Category {
            return strings.Compare(a.Category, b.Category)
        }
        if a.Price < b.Price {
            return -1
        }
        if a.Price > b.Price {
            return 1
        }
        return 0
    }),
)

subscription := observable.Subscribe(ro.NewObserver(
    func(product Product) {
        fmt.Printf("%s - %s: $%.2f\n", product.Category, product.Name, product.Price)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))

// Output:
// Books - Book: $19.99
// Electronics - Phone: $699.99
// Electronics - Laptop: $999.99
// Office - Pen: $2.99
// Completed
```

### Combining with Other Operators

```go
import (
    "github.com/samber/ro"
    rosort "github.com/samber/ro/plugins/sort"
)

// Generate random numbers, filter even ones, sort them
observable := ro.Pipe1(
    ro.Just(3, 1, 4, 1, 5, 9, 2, 6, 8, 7),
    ro.Filter(func(n int) bool {
        return n%2 == 0 // Keep only even numbers
    }),
    rosort.Sort[int](),
)

subscription := observable.Subscribe(ro.NewObserver(
    func(value int) {
        fmt.Printf("%d ", value)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("\nCompleted")
    },
))

// Output:
// 2 4 6 8
// Completed
```

## Performance Considerations

⚠️ **Important Warning**: These sorting operators have significant limitations:

1. **Memory Usage**: All values are loaded into memory before sorting
2. **Blocking**: The entire stream must complete before any sorted values are emitted
3. **Not Reactive**: This breaks the reactive programming paradigm

### When to Use

- Small datasets (< 1000 items)
- When you need sorted results for the entire stream
- Development and testing scenarios
- When memory usage is not a concern

### When NOT to Use

- Large datasets (> 1000 items)
- Real-time streaming applications
- Memory-constrained environments
- When you need immediate results

### Alternatives

For large datasets or real-time applications, consider:

1. **Database sorting**: Sort at the database level
2. **Streaming sort**: Use specialized streaming sort algorithms
3. **Chunked processing**: Process data in smaller chunks
4. **External sorting**: Use external sort utilities

## Error Handling

The sort operators handle errors gracefully:

```go
observable := ro.Pipe1(
    ro.Just(3, 1, 4, 1, 5),
    rosort.Sort[int](),
)

subscription := observable.Subscribe(ro.NewObserver(
    func(value int) {
        fmt.Printf("%d ", value)
    },
    func(err error) {
        // Handle errors (e.g., memory allocation failed)
        fmt.Printf("Sorting error: %v\n", err)
    },
    func() {
        fmt.Println("\nCompleted")
    },
))
```

## Dependencies

This plugin uses Go's standard library sorting:

- `slices.Sort` for ordered types
- `slices.SortFunc` for custom comparison
- `slices.SortStableFunc` for stable sorting

No additional dependencies are required. 