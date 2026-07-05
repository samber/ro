---
name: HTTPRequestJSON
slug: httprequestjson
sourceRef: plugins/http/client/source.go#L55
type: plugin
category: http-client
signatures:
  - "func HTTPRequestJSON[T any](req *http.Request, client *http.Client)"
playUrl: ""
variantHelpers:
  - plugin#http-client#httprequestjson
similarHelpers:
  - plugin#http-client#httprequest
position: 10
---

Sends HTTP requests and automatically decodes JSON responses.

```go
import (
    "fmt"
    "net/http"

    "github.com/samber/ro"
    rohttp "github.com/samber/ro/plugins/http/client"
)

type User struct {
    ID    int    `json:"id"`
    Name  string `json:"name"`
    Email string `json:"email"`
}

req, _ := http.NewRequest("GET", "https://api.example.com/users/1", nil)
obs := rohttp.HTTPRequestJSON[User](req, nil)

sub := obs.Subscribe(ro.NewObserver(
    func(user User) {
        fmt.Printf("User: {ID:%d Name:%s Email:%s}\n", user.ID, user.Name, user.Email)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// User: {ID:1 Name:Alice Email:alice@example.com}
// Completed
```