---
name: HTTPRequest
slug: httprequest
sourceRef: plugins/http/client/source.go#L30
type: plugin
category: http-client
signatures:
  - "func HTTPRequest(req *http.Request, client *http.Client)"
playUrl: ""
variantHelpers:
  - plugin#http-client#httprequest
similarHelpers:
  - plugin#http-client#httprequestjson
position: 0
---

Sends HTTP requests and returns the response.

```go
import (
    "fmt"
    "net/http"

    "github.com/samber/ro"
    rohttp "github.com/samber/ro/plugins/http/client"
)

req, _ := http.NewRequest("GET", "https://api.example.com/users", nil)
obs := rohttp.HTTPRequest(req, nil)

sub := obs.Subscribe(ro.NewObserver(
    func(resp *http.Response) {
        defer resp.Body.Close()
        fmt.Printf("Status: %s\n", resp.Status)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Status: 200 OK
// Completed
```