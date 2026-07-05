---
name: NewWebsocketSubject
slug: newwebsocketsubject
sourceRef: plugins/websocket/client/subject.go#L46
type: plugin
category: websocket-client
signatures:
  - "func NewWebsocketSubject[In any, Out any](config WebsocketSubjectConfig[In, Out]) *websocketSubject[In, Out]"
playUrl: ""
variantHelpers:
  - plugin#websocket-client#newwebsocketsubject
similarHelpers:
  - plugin#websocket-client#newwebsocketobservable
  - plugin#websocket-client#newwebsocketobserver
position: 2
---

Creates a WebSocket subject that can both send and receive data from a WebSocket endpoint.

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

config := rowebsocket.WebsocketSubjectConfig[Message, Message]{
    URL: "ws://localhost:8080/ws",
    Headers: map[string]string{
        "Authorization": "Bearer token123",
    },
    Serializer: func(msg Message) ([]byte, error) {
        return json.Marshal(msg)
    },
    Deserializer: func(data []byte) (Message, error) {
        var msg Message
        err := json.Unmarshal(data, &msg)
        return msg, err
    },
    Dialer: websocket.DefaultDialer,
    OutputConnector: func() ro.Subject[Message] {
        return ro.NewPublishSubject[Message]()
    },
}

subject := rowebsocket.NewWebsocketSubject(config)

// Subscribe to receive messages
sub := subject.Subscribe(ro.NewObserver(
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

// Send messages
subject.Next(Message{Text: "Hello server!"})
subject.Next(Message{Text: "Another message"})

// Received: Message from server
// Received: Response to message
```