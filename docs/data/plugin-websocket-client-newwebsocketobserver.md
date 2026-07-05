---
name: NewWebsocketObserver
slug: newwebsocketobserver
sourceRef: plugins/websocket/client/observer.go#L30
type: plugin
category: websocket-client
signatures:
  - "func NewWebsocketObserver[In any](config WebsocketObserverConfig[In]) ro.Observer[In]"
playUrl: ""
variantHelpers:
  - plugin#websocket-client#newwebsocketobserver
similarHelpers:
  - plugin#websocket-client#newwebsocketobservable
  - plugin#websocket-client#newwebsocketsubject
position: 1
---

Creates a WebSocket observer for sending data to a WebSocket endpoint.

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

config := rowebsocket.WebsocketObserverConfig[Message]{
    URL: "ws://localhost:8080/ws",
    Headers: map[string]string{
        "Authorization": "Bearer token123",
    },
    Serializer: func(msg Message) ([]byte, error) {
        return json.Marshal(msg)
    },
    Dialer: websocket.DefaultDialer,
}

observer := rowebsocket.NewWebsocketObserver(config)

// Send messages
observer.Next(Message{Text: "Hello server!"})
observer.Next(Message{Text: "Another message"})

observer.Complete() // Close the connection
```