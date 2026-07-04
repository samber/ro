---
name: Serialize
slug: serialize
sourceRef: operator_utility.go#L634
type: core
category: utility
signatures:
  - "func Serialize[T any]()"
playUrl: https://go.dev/play/p/KcXb17qceLb
variantHelpers:
  - core#utility#serialize
similarHelpers: []
position: 110
---

Serialize ensures thread-safe message passing by wrapping any observable in a SafeObservable implementation. This is useful when you need guaranteed serialization in concurrent scenarios where multiple goroutines might emit to the same observer.

```go
import (
    "fmt"
    "sync"
    "github.com/samber/ro"
)

// Concurrent producer emitting from multiple goroutines (unsafe without Serialize)
producer := ro.NewUnsafeObservable(func(observer ro.Observer[int]) ro.Teardown {
    var wg sync.WaitGroup
    for i := 0; i < 5; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            observer.Next(id)
        }(i)
    }
    go func() {
        wg.Wait()
        observer.Complete()
    }()
    return nil
})

// Serialize wraps the unsafe observable in a thread-safe one
obs := ro.Pipe[int, int](
    producer,
    ro.Serialize[int](),
)

values, _ := ro.Collect(obs)
fmt.Printf("Received %d values without race conditions\n", len(values))

// Received 5 values without race conditions
```
