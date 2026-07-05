---
name: NewWebsocketObservable
slug: newwebsocketobservable
sourceRef: plugins/websocket/client/observable.go#L37
type: plugin
category: websocket-client
signatures:
  - "func NewWebsocketObservable[Out any](config WebsocketObservableConfig[struct{}, Out]) ro.Observable[Out]"
playUrl: ""
variantHelpers:
  - plugin#websocket-client#newwebsocketobservable
similarHelpers:
  - plugin#websocket-client#newwebsocketsubject
  - plugin#websocket-client#newwebsocketobserver
position: 0
---

Creates a WebSocket observable for receiving data from a WebSocket endpoint.

```go
import (
    "encoding/json"

    "github.com/gorilla/websocket"
    "github.com/samber/ro"
    rowebsocket "github.com/samber/ro/plugins/websocket/client"
)

type Message struct {
    Text string `json:"text"`
}

config := rowebsocket.WebsocketObservableConfig[struct{}, Message]{
    URL: "ws://localhost:8080/ws",
    Headers: map[string]string{
        "Authorization": "Bearer token123",
    },
    Deserializer: func(data []byte) (Message, error) {
        var msg Message
        err := json.Unmarshal(data, &msg)
        return msg, err
    },
    Dialer: websocket.DefaultDialer,
}

obs := rowebsocket.NewWebsocketObservable(config)

sub := obs.Subscribe(ro.NewObserver(
    func(msg Message) {
        println("Received:", msg.Text)
    },
    func(err error) {
        println("Error:", err.Error())
    },
    func() {
        println("Connection closed")
    },
))
defer sub.Unsubscribe()

// Received: Hello from server!
// Received: Another message
```