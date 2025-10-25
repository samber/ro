---
name: ToMap
slug: tomap
sourceRef: operator_sink.go#L60
type: core
category: sink
signatures:
  - "func ToMap[T any, K comparable, V any](project func(item T) (K, V))"
  - "func ToMapWithContext[T any, K comparable, V any](project func(ctx context.Context, item T) (K, V))"
  - "func ToMapI[T any, K comparable, V any](mapper func(item T, index int64) (K, V))"
  - "func ToMapIWithContext[T any, K comparable, V any](mapper func(ctx context.Context, item T, index int64) (K, V))"
playUrl: https://go.dev/play/p/FiF83XYB0ba
variantHelpers:
  - core#sink#tomap
  - core#sink#tomapwithcontext
  - core#sink#tomapi
  - core#sink#tomapiwithcontext
similarHelpers:
  - core#sink#toslice
  - core#sink#tochannel
position: 20
---

Collects all emissions from the source Observable into a map. Items are keyed by the result of the key selector function.

```go
obs := ro.Pipe[string, map[string]string](
    ro.Just("apple", "banana", "cherry"),
    ro.ToMap(func(s string) (string, string) {
        return s[:1], s // Use first letter as key, whole string as value
    }),
)

sub := obs.Subscribe(ro.PrintObserver[map[string]string]())
defer sub.Unsubscribe()

// Next: map[a:apple b:banana c:cherry]
// Completed
```

### ToMapWithValue

```go
type User struct {
    id   int
    name string
}

obs := ro.Pipe[User, map[int]string](
    ro.Just(
        User{id: 1, name: "Alice"},
        User{id: 2, name: "Bob"},
        User{id: 3, name: "Charlie"},
    ),
    ro.ToMapWithValue(
        func(u User) int { return u.id },
        func(u User) string { return u.name },
    ),
)

sub := obs.Subscribe(ro.PrintObserver[map[int]string]())
defer sub.Unsubscribe()

// Next: map[1:Alice 2:Bob 3:Charlie]
// Completed
```

### With key collisions (last value wins)

```go
obs := ro.Pipe[string, map[string]string](
    ro.Just("apple", "avocado", "banana"),
    ro.ToMap(func(s string) string {
        return s[:1] // 'a' appears twice
    }),
)

sub := obs.Subscribe(ro.PrintObserver[map[string]string]())
defer sub.Unsubscribe()

// Next: map[a:avocado b:banana]
// Completed (avocado overwrites apple)
```

### With empty observable

```go
obs := ro.Pipe[string, map[int]string](
    ro.Empty[string](),
    ro.ToMap(func(s string) int {
        return len(s)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[map[int]string]())
defer sub.Unsubscribe()

// Next: map[]
// Completed
```

### With error handling

```go
obs := ro.Pipe[int, map[int]int](
    ro.Pipe[int, int](
        ro.Just(1, 2, 3),
        ro.MapErr(func(i int) (int, error) {
            if i == 3 {
                return 0, fmt.Errorf("error on 3")
            }
            return i, nil
        }),
    ),
    ro.ToMap(func(i int) int {
        return i
    }),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value map[int]int) {
        fmt.Printf("Map: %v\n", value)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Complete")
    },
))
defer sub.Unsubscribe()

// Error: error on 3
// (No map emitted due to error)
```

### With duplicate keys handling

```go
// Example showing how to handle collisions by creating composite values
obs := ro.Pipe[string, map[string][]string](
    ro.Just(
        "file1.txt",
        "file2.txt",
        "image1.jpg",
        "image2.jpg",
        "doc1.pdf",
    ),
    ro.ToMapWithValue(
        func(s string) string {
            // Extract extension as key
            if dot := strings.LastIndex(s, "."); dot > 0 {
                return s[dot+1:]
            }
            return "unknown"
        },
        func(s string) []string {
            // Collect all files for each extension
            return []string{s}
        },
    ),
)

sub := obs.Subscribe(ro.PrintObserver[map[string][]string]())
defer sub.Unsubscribe()

// Next: map[jpg:[image1.jpg image2.jpg] pdf:[doc1.pdf] txt:[file1.txt file2.txt]]
// This won't work as expected since values get overwritten.
// See next example for proper collision handling.
```

### With complex value aggregation

```go
// For true duplicate handling, preprocess the data first
type FileGroup struct {
    Extension string
    Files     []string
}

obs := ro.Pipe[string, map[string][]string](
    ro.Just("file1.txt", "file2.txt", "image1.jpg", "image2.jpg"),
    ro.ToSlice[string](),
    ro.Map(func(files []string) map[string][]string {
        result := make(map[string][]string)
        for _, file := range files {
            if dot := strings.LastIndex(file, "."); dot > 0 {
                ext := file[dot+1:]
                result[ext] = append(result[ext], file)
            }
        }
        return result
    }),
)

sub := obs.Subscribe(ro.PrintObserver[map[string][]string]())
defer sub.Unsubscribe()

// Next: map[jpg:[image1.jpg image2.jpg] txt:[file1.txt file2.txt]]
// Completed
```

### With struct transformation

```go
type Product struct {
    SKU   string
    Name  string
    Price float64
}

obs := ro.Pipe[Product, map[string]Product](
    ro.Just(
        Product{SKU: "P001", Name: "Laptop", Price: 999.99},
        Product{SKU: "P002", Name: "Mouse", Price: 29.99},
        Product{SKU: "P003", Name: "Keyboard", Price: 79.99},
    ),
    ro.ToMapWithValue(
        func(p Product) string { return p.SKU },
        func(p Product) Product { return p },
    ),
)

sub := obs.Subscribe(ro.PrintObserver[map[string]Product]())
defer sub.Unsubscribe()

// Next: map[P001:{SKU:P001 Name:Laptop Price:999.99} P002:{SKU:P002 Name:Mouse Price:29.99} P003:{SKU:P003 Name:Keyboard Price:79.99}]
// Completed
```

### With hot observable

```go
source := ro.Interval(100 * time.Millisecond)
obs := ro.Pipe[int64, map[int]string](
    source,
    ro.Take[int64](5),
    ro.ToMapWithValue(
        func(i int64) int { return int(i) },
        func(i int64) string { return fmt.Sprintf("item-%d", i) },
    ),
)

sub := obs.Subscribe(ro.PrintObserver[map[int]string]())
time.Sleep(700 * time.Millisecond)
sub.Unsubscribe()

// Next: map[0:item-0 1:item-1 2:item-2 3:item-3 4:item-4]
// Completed
```

### With filtered data

```go
obs := ro.Pipe[int, map[int]string](
    ro.Range(1, 20),
    ro.Filter(func(i int) bool {
        return i%3 == 0 // Only multiples of 3
    }),
    ro.ToMapWithValue(
        func(i int) int { return i },
        func(i int) string { return fmt.Sprintf("multiple-%d", i) },
    ),
)

sub := obs.Subscribe(ro.PrintObserver[map[int]string]())
defer sub.Unsubscribe()

// Next: map[3:multiple-3 6:multiple-6 9:multiple-9 12:multiple-12 15:multiple-15 18:multiple-18]
// Completed
```

### With async operations

```go
obs := ro.Pipe[struct {
    ID   string
    Data string
}, map[string]string](
    ro.Pipe[string, struct {
        ID   string
        Data string
    }](
        ro.Just("user1", "user2", "user3"),
        MapAsync(func(userID string) Observable[struct {
            ID   string
            Data string
        }] {
            return Defer(func() Observable[struct {
                ID   string
                Data string
            }] {
                time.Sleep(50 * time.Millisecond)
                return ro.Just(struct {
                    ID   string
                    Data string
                }{
                    ID:   userID,
                    Data: "data_for_" + userID,
                })
            })
        }, 2),
    ),
    ro.ToMapWithValue(
        func(user struct {
            ID   string
            Data string
        }) string { return user.ID },
        func(user struct {
            ID   string
            Data string
        }) string { return user.Data },
    ),
)

sub := obs.Subscribe(ro.PrintObserver[map[string]string]())
time.Sleep(300 * time.Millisecond)
defer sub.Unsubscribe()

// Next: map[user1:data_for_user1 user2:data_for_user2 user3:data_for_user3]
// Completed
```

### With real-time data collection

```go
// Simulate sensor data collection by sensor ID
sensorReadings := ro.Interval(500 * time.Millisecond)
obs := ro.Pipe[struct {
    SensorID string
    Value    float64
    Time     int64
}, map[string]float64](
    sensorReadings,
    ro.Take[int64](10),
    ro.Map(func(timestamp int64) struct {
        SensorID string
        Value    float64
        Time     int64
    } {
        sensors := []string{"temp-01", "temp-02", "humidity-01"}
        sensor := sensors[int(timestamp)%len(sensors)]
        return struct {
            SensorID string
            Value    float64
            Time     int64
        }{
            SensorID: sensor,
            Value:    20.0 + rand.Float64()*15,
            Time:     time.Now().Unix(),
        }
    }),
    ro.ToMapWithValue(
        func(reading struct {
            SensorID string
            Value    float64
            Time     int64
        }) string { return reading.SensorID },
        func(reading struct {
            SensorID string
            Value    float64
            Time     int64
        }) float64 { return reading.Value },
    ),
)

sub := obs.Subscribe(ro.OnNext(func(readings map[string]float64) {
    fmt.Printf("Latest sensor readings: %v\n", readings)
}))
time.Sleep(6 * time.Second)
sub.Unsubscribe()
```