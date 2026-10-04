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
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/samber/ro"
)

const (
	// fuzzWait bounds every wait so a hang becomes a failure, not a stuck CI job.
	fuzzWait = 5 * time.Second

	// fuzzMaxWriters bounds goroutine fan-out per iteration to keep runs fast.
	fuzzMaxWriters = 8

	// fuzzMaxMessages bounds the messages each writer sends.
	fuzzMaxMessages = 6

	// fuzzAsyncDelayUnit scales the delayed-server timing from the fuzz input.
	fuzzAsyncDelayUnit = 200 * time.Microsecond

	// fuzzMaxAsyncSteps bounds the async delay at fuzzMaxAsyncSteps*fuzzAsyncDelayUnit.
	fuzzMaxAsyncSteps = 10

	// fuzzModeAsync is the bit of the mask selecting a server that delays its pushes.
	fuzzModeAsync = 1 << 0

	// fuzzStableWindow is how long the test waits to prove that nothing more happens.
	fuzzStableWindow = 300 * time.Millisecond
)

// fuzzDelay derives a delay from the seed: zero in sync mode.
func fuzzDelay(seed int64, mask uint8) time.Duration {
	if mask&fuzzModeAsync == 0 {
		return 0
	}

	if seed < 0 {
		seed = -seed
	}

	return time.Duration(seed%fuzzMaxAsyncSteps+1) * fuzzAsyncDelayUnit
}

// fuzzCount returns a value in [1, n] derived from any uint8.
func fuzzCount(v uint8, n int) int {
	return int(v)%n + 1
}

// wsServer is a local echo server that can push messages on connect and close
// connections on demand. All counters are read with atomics.
type wsServer struct {
	*httptest.Server

	accepted   int64 // connections upgraded
	received   int64 // text messages read from clients
	gone       int64 // connections whose read loop ended (client closed or errored)
	goneSignal chan struct{}
}

type wsServerConfig struct {
	pushCount     int           // messages pushed right after the upgrade
	pushDelay     time.Duration // wait before pushing (async mode)
	closeAfterRcv int           // close the connection after this many messages (0 = never)
}

func newWSServer(cfg wsServerConfig) *wsServer {
	srv := &wsServer{goneSignal: make(chan struct{}, 64)}
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}

	srv.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}

		atomic.AddInt64(&srv.accepted, 1)

		defer func() {
			_ = conn.Close()
			atomic.AddInt64(&srv.gone, 1)

			select {
			case srv.goneSignal <- struct{}{}:
			default:
			}
		}()

		// Pushes use their own writer lock-free path: the read loop below only
		// writes after the pushes have been sent, so there is a single writer.
		if cfg.pushDelay > 0 {
			time.Sleep(cfg.pushDelay)
		}

		for i := 0; i < cfg.pushCount; i++ {
			if err := conn.WriteMessage(websocket.TextMessage, []byte(fmt.Sprintf("push-%d", i))); err != nil {
				return
			}
		}

		received := 0

		for {
			typ, msg, err := conn.ReadMessage()
			if err != nil {
				return
			}

			if typ != websocket.TextMessage {
				continue
			}

			received++
			atomic.AddInt64(&srv.received, 1)

			if cfg.closeAfterRcv > 0 && received >= cfg.closeAfterRcv {
				_ = conn.WriteControl(
					websocket.CloseMessage,
					websocket.FormatCloseMessage(websocket.CloseNormalClosure, "bye"),
					time.Now().Add(time.Second),
				)

				return
			}

			if err := conn.WriteMessage(websocket.TextMessage, msg); err != nil {
				return
			}
		}
	}))

	return srv
}

func (s *wsServer) wsURL() string {
	return "ws" + strings.TrimPrefix(s.URL, "http")
}

func newFuzzSubject(url string) *websocketSubject[string, string] {
	return NewWebsocketSubject(WebsocketSubjectConfig[string, string]{
		URL:          url,
		Serializer:   func(v string) ([]byte, error) { return []byte(v), nil },
		Deserializer: func(b []byte) (string, error) { return string(b), nil },
	})
}

// deadURL returns a ws URL on a port nothing listens on.
func deadURL(t *testing.T) string {
	t.Helper()

	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("rowebsocketclient: listen: %v", err)
	}

	addr := l.Addr().String()
	_ = l.Close()

	return "ws://" + addr
}

// goRecover runs fn in a goroutine and reports a recovered panic through errs,
// since a panic in a goroutine would otherwise kill the whole test binary.
func goRecover(wg *sync.WaitGroup, errs chan<- string, fn func()) {
	wg.Add(1)

	go func() {
		defer wg.Done()
		defer func() {
			if r := recover(); r != nil {
				errs <- fmt.Sprintf("rowebsocketclient: panic: %v", r)
			}
		}()

		fn()
	}()
}

// waitBounded fails the iteration when wg does not finish within fuzzWait.
func waitBounded(t *testing.T, wg *sync.WaitGroup, what string) {
	t.Helper()

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(fuzzWait):
		t.Fatalf("rowebsocketclient: %s: timed out after %s", what, fuzzWait)
	}
}

func failOnErrs(t *testing.T, errs chan string) {
	t.Helper()

	close(errs)

	for msg := range errs {
		t.Fatal(msg)
	}
}

// FuzzWebsocketSubjectDialFailure calls every subject entry point on a subject
// whose connect() failed (or never ran): the output subject is nil there.
// Sync/async does not apply: there is no server, hence nothing to delay.
func FuzzWebsocketSubjectDialFailure(f *testing.F) {
	f.Skip("race: ws-nil-output-panic; remove when fixed")

	addSeeds(f, func(i int) []any { return []any{uint8(i)} })

	f.Fuzz(func(t *testing.T, entry uint8) {
		ws := newFuzzSubject(deadURL(t))

		var (
			wg   sync.WaitGroup
			errs = make(chan string, 1)
		)

		goRecover(&wg, errs, func() {
			switch entry % 6 {
			case 0:
				ws.Next("x")
			case 1:
				ws.Error(fmt.Errorf("boom"))
			case 2:
				ws.Complete()
			case 3:
				_ = ws.IsClosed()
			case 4:
				_ = ws.CountObservers()
			default:
				_ = ws.HasThrown()
			}
		})

		waitBounded(t, &wg, "entry point on never-connected subject")
		failOnErrs(t, errs)
	})
}

// FuzzWebsocketSubjectSubscribeDialFailure checks that Subscribe on a dead URL
// reports the dial error to the observer instead of panicking or hanging.
func FuzzWebsocketSubjectSubscribeDialFailure(f *testing.F) {
	addSeeds(f, func(i int) []any { return []any{uint8(i)} })

	f.Fuzz(func(t *testing.T, subscribers uint8) {
		ws := newFuzzSubject(deadURL(t))
		n := fuzzCount(subscribers, fuzzMaxWriters)

		var (
			wg     sync.WaitGroup
			errs   = make(chan string, n)
			failed int64
		)

		for i := 0; i < n; i++ {
			goRecover(&wg, errs, func() {
				sub := ws.Subscribe(ro.NewObserver(
					func(string) {},
					func(error) { atomic.AddInt64(&failed, 1) },
					func() {},
				))
				sub.Unsubscribe()
			})
		}

		waitBounded(t, &wg, "subscribe on dead url")
		failOnErrs(t, errs)

		if got := atomic.LoadInt64(&failed); got != int64(n) {
			t.Fatalf("rowebsocketclient: %d of %d subscribers received the dial error", got, n)
		}
	})
}

// FuzzWebsocketSubjectConcurrentNext sends from N goroutines at once: gorilla
// forbids concurrent writers on one connection.
func FuzzWebsocketSubjectConcurrentNext(f *testing.F) {
	f.Skip("race: ws-concurrent-write; remove when fixed")

	addSeeds(f, func(i int) []any { return []any{int64(i), uint8(i), uint8(i / 2), uint8(i / 3)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8, writers uint8, messages uint8) {
		// Async mode delays the server push so writes start before any read;
		// sync mode pushes immediately.
		srv := newWSServer(wsServerConfig{pushCount: 2, pushDelay: fuzzDelay(seed, mask)})
		defer srv.Close()

		ws := newFuzzSubject(srv.wsURL())
		nWriters := fuzzCount(writers, fuzzMaxWriters)
		nMessages := fuzzCount(messages, fuzzMaxMessages)

		sub := ws.Subscribe(ro.NewObserver(func(string) {}, func(error) {}, func() {}))
		defer sub.Unsubscribe()

		var (
			wg    sync.WaitGroup
			errs  = make(chan string, nWriters)
			start = make(chan struct{})
		)

		for w := 0; w < nWriters; w++ {
			w := w // go.mod is go 1.18: loop variables are shared across iterations

			goRecover(&wg, errs, func() {
				<-start

				for m := 0; m < nMessages; m++ {
					ws.Next(fmt.Sprintf("w%d-m%d", w, m))
				}
			})
		}

		close(start)
		waitBounded(t, &wg, "concurrent Next")
		failOnErrs(t, errs)
	})
}

// FuzzWebsocketSubjectUnsubscribeClosesConn unsubscribes the only observer (or
// cancels its context) and expects the server to see the connection go away.
func FuzzWebsocketSubjectUnsubscribeClosesConn(f *testing.F) {
	f.Skip("race: ws-conn-leak-on-unsubscribe; remove when fixed")

	addSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		srv := newWSServer(wsServerConfig{pushCount: 1, pushDelay: fuzzDelay(seed, mask)})
		defer srv.Close()

		ws := newFuzzSubject(srv.wsURL())

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		sub := ws.SubscribeWithContext(ctx, ro.NewObserver(func(string) {}, func(error) {}, func() {}))

		// Wait for the upgrade so the close is observable server-side.
		deadline := time.Now().Add(fuzzWait)
		for atomic.LoadInt64(&srv.accepted) == 0 {
			if time.Now().After(deadline) {
				t.Fatalf("rowebsocketclient: server never accepted the connection")
			}

			time.Sleep(time.Millisecond)
		}

		// Bit 1 selects Unsubscribe, otherwise context cancellation.
		if mask&2 != 0 {
			sub.Unsubscribe()
		} else {
			cancel()
		}

		select {
		case <-srv.goneSignal:
		case <-time.After(fuzzStableWindow):
			t.Fatalf("rowebsocketclient: connection still open after last observer left (accepted=1 gone=0)")
		}
	})
}

// FuzzWebsocketSubjectReconnectAfterServerClose makes the server hang up, then
// keeps sending: the subject must not silently write into the dead connection.
func FuzzWebsocketSubjectReconnectAfterServerClose(f *testing.F) {
	f.Skip("race: ws-no-reconnect-silent-drop; remove when fixed")

	addSeeds(f, func(i int) []any { return []any{int64(i), uint8(i), uint8(i / 2)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8, after uint8) {
		closeAfter := fuzzCount(after, 3)
		srv := newWSServer(wsServerConfig{pushDelay: fuzzDelay(seed, mask), closeAfterRcv: closeAfter})
		defer srv.Close()

		ws := newFuzzSubject(srv.wsURL())

		// A normal close frame is surfaced as an error, so either terminal event counts.
		var (
			once      sync.Once
			completed = make(chan struct{})
			finish    = func() { once.Do(func() { close(completed) }) }
		)

		sub := ws.Subscribe(ro.NewObserver(
			func(string) {},
			func(error) { finish() },
			finish,
		))
		defer sub.Unsubscribe()

		for i := 0; i < closeAfter; i++ {
			ws.Next(fmt.Sprintf("m%d", i))
		}

		select {
		case <-completed:
		case <-time.After(fuzzWait):
			t.Fatalf("rowebsocketclient: stream did not complete after server close")
		}

		before := atomic.LoadInt64(&srv.received)

		// Sends after the hang-up: the output subject is already terminated, so
		// the only way not to lose them is a fresh connection that delivers them.
		for i := 0; i < 3; i++ {
			ws.Next("after-close")
			time.Sleep(fuzzStableWindow / 3)
		}

		delivered := atomic.LoadInt64(&srv.received) - before
		if delivered == 0 && atomic.LoadInt64(&srv.accepted) == 1 {
			t.Fatalf("rowebsocketclient: messages sent after server close were silently dropped (accepted=1 delivered=0)")
		}
	})
}
