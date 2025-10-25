---
name: Unmarshal
slug: unmarshal
sourceRef: plugins/encoding/json/operator.go#L30
type: plugin
category: encoding-json
signatures:
  - "func Unmarshal[T any]()"
playUrl: https://go.dev/play/p/aMiYMUUkjnt
variantHelpers:
  - plugin#encoding-json#unmarshal
similarHelpers: 
  - plugin#encoding-json-v2#unmarshal
  - plugin#encoding-gob#decode
position: 10
---

Decodes JSON format to typed values.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rojson "github.com/samber/ro/plugins/encoding/json"
)

type User struct {
    ID   int    `json:"id"`
    Name string `json:"name"`
    Age  int    `json:"age"`
}

obs := ro.Pipe[[]byte, User](
    ro.Just([]byte(`{"id":1,"name":"Alice","age":30}`)),
    rojson.Unmarshal[User](),
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