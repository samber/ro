---
name: Marshal
slug: marshal
sourceRef: plugins/encoding/json/operator.go#L24
type: plugin
category: encoding-json
signatures:
  - "func Marshal[T any]()"
playUrl: https://go.dev/play/p/XUeF_VVg62I
variantHelpers:
  - plugin#encoding-json#marshal
similarHelpers:
  - plugin#encoding-json-v2#marshal
  - plugin#encoding-gob#encode
position: 0
---

Encodes values to JSON format.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rojson "github.com/samber/ro/plugins/encoding/json"
)

type User struct {
    ID   int
    Name string
    Age  int
}

obs := ro.Pipe[User, []byte](
    ro.Just(User{ID: 1, Name: "Alice", Age: 30}),
    rojson.Marshal[User](),
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