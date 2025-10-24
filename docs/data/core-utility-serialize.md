---
name: Serialize
slug: serialize
sourceRef: operator_utility.go#L634
type: core
category: utility
signatures:
  - "func Serialize[T any]()"
playUrl:
variantHelpers:
  - core#utility#serialize
similarHelpers: []
position: 110
---

Serialize ensures thread-safe message passing by wrapping any observable in a SafeObservable implementation. This is useful when you need guaranteed serialization in concurrent scenarios where multiple goroutines might emit to the same observer.

```go
import (
    "github.com/samber/ro"
)

// Concurrent producer that emits from multiple goroutines
func createConcurrentProducer() ro.Observable[int] {
    return ro.NewUnsafeObservable(func(observer ro.Observer[int]) ro.Teardown {
        for i := 0; i < 3; i++ {
            go func(id int) {
                for j := 0; j < 5; j++ {
                    value := id*10 + j
                    observer.Next(value) // Concurrent emissions
                }
            }(i)
        }

        time.Sleep(100 * time.Millisecond)
        observer.Complete()
        return nil
    })
}

// Serialize ensures thread-safe message passing
obs := ro.Pipe2(
    createConcurrentProducer(),
    ro.Serialize[int](), // Wraps in safe observable for serialization
    ro.Distinct[int](),  // Distinct operator is not protected against race conditions
)

sub := obs.Subscribe(ro.OnNext(func(value int) {
    fmt.Printf("Received: %d\n", value)
}))
defer sub.Unsubscribe()

// expected result: Values 0-14 received in sequential order without race conditions
```
