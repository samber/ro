// Copyright 2025 samber.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://github.com/samber/ro/blob/main/licenses/LICENSE.apache.md
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package rowebsocketclient

import (
	"context"
	"net/http"
	"sync"

	"github.com/gorilla/websocket"
	"github.com/samber/ro"
)

type Serializer[T any] func(T) ([]byte, error)

type Deserializer[T any] func([]byte) (T, error)

type WebsocketSubjectConfig[In any, Out any] struct {
	URL          string
	Headers      map[string]string
	Serializer   Serializer[In]
	Deserializer Deserializer[Out]
	Dialer       *websocket.Dialer

	// Connector is a function that returns a Subject[Out].
	// This is useful when you want to use a different Subject implementation.
	// For example, you could use a ReplaySubject to replay the last N messages.
	OutputConnector func() ro.Subject[Out]
	// ResetOnError    bool
	// ResetOnComplete bool
}

// NewWebsocketSubject creates a websocket subject that can both send and receive messages from a websocket endpoint.
func NewWebsocketSubject[In any, Out any](config WebsocketSubjectConfig[In, Out]) *websocketSubject[In, Out] {
	if config.URL == "" {
		panic(ErrWebsocketSubjectURLRequired)
	}
	if config.Serializer == nil {
		panic(ErrWebsocketSubjectSerializerRequired)
	}
	if config.Deserializer == nil {
		panic(ErrWebsocketSubjectDeserializerRequired)
	}
	if config.Dialer == nil {
		config.Dialer = websocket.DefaultDialer
	}

	// Set default output connector
	if config.OutputConnector == nil {
		config.OutputConnector = func() ro.Subject[Out] {
			return ro.NewPublishSubject[Out]()
		}
	}

	return &websocketSubject[In, Out]{
		config: config,
		output: nil,
	}
}

var _ ro.Subject[string] = (*websocketSubject[string, string])(nil)
var _ ro.Observer[string] = (*websocketSubject[string, int])(nil)
var _ ro.Observable[string] = (*websocketSubject[int, string])(nil)

type websocketSubject[In any, Out any] struct {
	config WebsocketSubjectConfig[In, Out]
	input  ro.Observer[[]byte]
	output ro.Subject[Out]
	conn   *websocket.Conn
	mu     sync.RWMutex
}

// Implements ro.Observable[Out]
func (ws *websocketSubject[In, Out]) Subscribe(destination ro.Observer[Out]) ro.Subscription {
	return ws.SubscribeWithContext(context.Background(), destination)
}

// Implements ro.Observable[Out]
func (ws *websocketSubject[In, Out]) SubscribeWithContext(ctx context.Context, destination ro.Observer[Out]) ro.Subscription {
	_, output, err := ws.connect()
	if err != nil {
		destination.ErrorWithContext(context.TODO(), err)
		sub := ro.NewSubscription(nil)
		sub.Unsubscribe()
		return sub
	}

	return output.SubscribeWithContext(ctx, destination)
}

// Implements ro.Observer[In]
func (ws *websocketSubject[In, Out]) Next(value In) {
	ws.NextWithContext(context.Background(), value)
}

// Implements ro.Observer[In]
func (ws *websocketSubject[In, Out]) NextWithContext(ctx context.Context, value In) {
	input, _, err := ws.connect()
	if err != nil {
		ws.ErrorWithContext(ctx, err)
		return
	}

	data, err := ws.config.Serializer(value)
	if err != nil {
		ws.ErrorWithContext(ctx, err)
		return
	}

	input.NextWithContext(ctx, data)
}

// Implements ro.Observer[In]
func (ws *websocketSubject[In, Out]) Error(err error) {
	ws.ErrorWithContext(context.Background(), err)
}

// Implements ro.Observer[In]
func (ws *websocketSubject[In, Out]) ErrorWithContext(ctx context.Context, err error) {
	ws.output.ErrorWithContext(ctx, err)
}

// Implements ro.Observer[In]
func (ws *websocketSubject[In, Out]) Complete() {
	ws.CompleteWithContext(context.Background())
}

// Implements ro.Observer[In]
func (ws *websocketSubject[In, Out]) CompleteWithContext(ctx context.Context) {
	ws.output.CompleteWithContext(ctx)
}

// Implements ro.Observer[In]
func (ws *websocketSubject[In, Out]) IsClosed() bool {
	return ws.output.IsClosed()
}

// Implements ro.Observer[In]
func (ws *websocketSubject[In, Out]) HasThrown() bool {
	return ws.output.HasThrown()
}

// Implements ro.Observer[In]
func (ws *websocketSubject[In, Out]) IsCompleted() bool {
	return ws.output.IsCompleted()
}

// Implements ro.Subject[Out]
func (ws *websocketSubject[In, Out]) HasObserver() bool {
	return ws.output.HasObserver()
}

// Implements ro.Subject[Out]
func (ws *websocketSubject[In, Out]) CountObservers() int {
	return ws.output.CountObservers()
}

// Implements ro.Subject[Out]
func (ws *websocketSubject[In, Out]) AsObservable() ro.Observable[Out] {
	return ws
}

// Implements ro.Subject[In]
func (ws *websocketSubject[In, Out]) AsObserver() ro.Observer[In] {
	return ws
}

// Connect establishes the WebSocket connection
func (ws *websocketSubject[In, Out]) connect() (ro.Observer[[]byte], ro.Subject[Out], error) {
	ws.mu.Lock()
	defer ws.mu.Unlock()

	if ws.conn != nil && ws.input != nil && ws.output != nil {
		return ws.input, ws.output, nil // Already connected
	}

	// Set up headers
	headers := http.Header{}
	for key, value := range ws.config.Headers {
		headers.Set(key, value)
	}

	// Dial the WebSocket connection
	conn, _, err := ws.config.Dialer.Dial(ws.config.URL, headers)
	if err != nil {
		return nil, nil, err
	}

	output := ws.config.OutputConnector()
	input := ro.NewObserverWithContext(
		func(ctx context.Context, value []byte) {
			// conn.SetWriteDeadline(time.Now().Add(?))
			err := ws.conn.WriteMessage(websocket.TextMessage, value)
			if err != nil {
				output.ErrorWithContext(ctx, err)
			}
		},
		func(ctx context.Context, err error) {
			output.ErrorWithContext(ctx, err)
			ws.conn.Close()
			ws.mu.Lock()
			ws.conn = nil
			ws.input = nil
			ws.output = nil
			ws.mu.Unlock()
		},
		func(ctx context.Context) {
			output.CompleteWithContext(ctx)
			ws.conn.Close()

			ws.mu.Lock()
			ws.conn = nil
			ws.input = nil
			ws.output = nil
			ws.mu.Unlock()
		},
	)

	ws.conn = conn
	ws.input = input
	ws.output = output

	// Start reading messages
	go ws.readMessages(ws.conn, ws.output)

	return ws.input, ws.output, nil
}

func (ws *websocketSubject[In, Out]) readMessages(conn *websocket.Conn, output ro.Subject[Out]) {
	defer output.CompleteWithContext(context.TODO())

	conn.SetPongHandler(func(string) error {
		return nil
	})

	for {
		messageType, message, err := ws.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				output.ErrorWithContext(context.TODO(), err)
			}
			return
		}

		if messageType == websocket.CloseMessage {
			break
		}

		if messageType != websocket.TextMessage {
			continue
		}

		// Deserialize and emit the message
		value, err := ws.config.Deserializer(message)
		if err != nil {
			output.ErrorWithContext(context.TODO(), err)
			continue
		}

		output.NextWithContext(context.TODO(), value)
	}
}
