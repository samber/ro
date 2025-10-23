---
name: Marshal
slug: marshal
sourceRef: plugins/encoding/json/v2/operator.go#L25
type: plugin
category: encoding-json-v2
signatures:
  - "func Marshal[T any]()"
playUrl: ""
variantHelpers:
  - plugin#encoding-json-v2#marshal
similarHelpers:
  - plugin#encoding-json#marshal
  - plugin#encoding-gob#encode
position: 20
---

Encodes values to JSON format using json/v2 (Go 1.25+).

```go
import (
    "fmt"

    "github.com/samber/ro"
    rojsonv2 "github.com/samber/ro/plugins/encoding/json/v2"
)

obs := ro.Pipe[User, []byte](
    ro.Just(User{ID: 1, Name: "Alice", Age: 30}),
    rojsonv2.Marshal[User](),
)

sub := obs.Subscribe(ro.NewObserver(
    func(data []byte) {
        fmt.Printf("Next: %s\n", string(data))
    },
    func(err error) {
        fmt.Printf("Error: %s\n", err.Error())
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: {"id":1,"name":"Alice","age":30}
// Completed
```