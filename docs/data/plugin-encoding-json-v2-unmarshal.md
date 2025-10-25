---
name: Unmarshal
slug: unmarshal
sourceRef: plugins/encoding/json/v2/operator.go#L31
type: plugin
category: encoding-json-v2
signatures:
  - "func Unmarshal[T any]()"
playUrl: https://go.dev/play/p/X8yk6QLdDw5
variantHelpers:
  - plugin#encoding-json-v2#unmarshal
similarHelpers:
  - plugin#encoding-json#unmarshal
  - plugin#encoding-gob#decode
position: 30
---

Decodes JSON format to typed values using json/v2 (Go 1.25+).

```go
import (
    "fmt"

    "github.com/samber/ro"
    rojsonv2 "github.com/samber/ro/plugins/encoding/json/v2"
)

obs := ro.Pipe[[]byte, User](
    ro.Just([]byte(`{"id":1,"name":"Alice","age":30}`)),
    rojsonv2.Unmarshal[User](),
)

sub := obs.Subscribe(ro.NewObserver(
    func(user User) {
        fmt.Printf("Next: {ID:%d Name:%s Age:%d}\n", user.ID, user.Name, user.Age)
    },
    func(err error) {
        fmt.Printf("Error: %s\n", err.Error())
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: {ID:1 Name:Alice Age:30}
// Completed
```