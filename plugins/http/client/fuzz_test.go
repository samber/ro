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

package rohttpclient

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/ro"
)

const (
	// fuzzWait bounds every wait so a hang becomes a failure, not a stuck CI job.
	fuzzWait = 5 * time.Second

	// fuzzMaxSubscribers bounds goroutine fan-out per iteration to keep runs fast.
	fuzzMaxSubscribers = 8

	// fuzzAsyncDelayUnit scales the async handler delay from the fuzz input.
	fuzzAsyncDelayUnit = 200 * time.Microsecond

	// fuzzMaxAsyncSteps bounds the async delay at fuzzMaxAsyncSteps*fuzzAsyncDelayUnit.
	fuzzMaxAsyncSteps = 10

	// fuzzCancelWindow is far above a loopback abort (sub-millisecond) yet short
	// enough that 100+ failing iterations stay under the test timeout.
	fuzzCancelWindow = 300 * time.Millisecond

	// fuzzModeAsync is the bit of the mask selecting a delayed (async) server.
	fuzzModeAsync = 1 << 0
)

// fuzzDelay derives the server-side delay from the seed: zero in sync mode.
func fuzzDelay(seed int64, mask uint8) time.Duration {
	if mask&fuzzModeAsync == 0 {
		return 0
	}

	if seed < 0 {
		seed = -seed
	}

	return time.Duration(seed%fuzzMaxAsyncSteps+1) * fuzzAsyncDelayUnit
}

// goRecover runs fn in a goroutine and reports a recovered panic through errs,
// since a panic in a goroutine would otherwise kill the whole test binary.
func goRecover(wg *sync.WaitGroup, errs chan<- string, fn func()) {
	wg.Add(1)

	go func() {
		defer wg.Done()
		defer func() {
			if r := recover(); r != nil {
				errs <- fmt.Sprintf("rohttpclient: panic: %v", r)
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
		t.Fatalf("rohttpclient: %s: timed out after %s", what, fuzzWait)
	}
}

func newFuzzServer(delay time.Duration, body string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if delay > 0 {
			select {
			case <-time.After(delay):
			case <-r.Context().Done():
				return
			}
		}

		_, _ = io.WriteString(w, body)
	}))
}

// FuzzHTTPRequestSharedObservable subscribes N goroutines to the SAME
// observable: HTTPRequest reassigns the captured req inside each subscription's
// goroutine, so concurrent subscriptions write a shared variable.
func FuzzHTTPRequestSharedObservable(f *testing.F) {
	f.Skip("race: http-shared-req; remove when fixed")

	addSeeds(f, func(i int) []any { return []any{int64(i), uint8(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8, subscribers uint8) {
		srv := newFuzzServer(fuzzDelay(seed, mask), "ok")
		defer srv.Close()

		req, err := http.NewRequest(http.MethodGet, srv.URL, nil)
		if err != nil {
			t.Fatalf("rohttpclient: new request: %v", err)
		}

		obs := HTTPRequest(req, srv.Client())
		n := int(subscribers)%fuzzMaxSubscribers + 1

		var (
			wg     sync.WaitGroup
			errs   = make(chan string, n)
			failed int64
		)

		start := make(chan struct{})

		for i := 0; i < n; i++ {
			goRecover(&wg, errs, func() {
				<-start

				done := make(chan struct{})
				sub := obs.Subscribe(ro.NewObserver(
					func(res *http.Response) { _ = res.Body.Close() },
					func(error) {
						atomic.AddInt64(&failed, 1)
						close(done)
					},
					func() { close(done) },
				))
				defer sub.Unsubscribe()

				select {
				case <-done:
				case <-time.After(fuzzWait):
					errs <- "rohttpclient: subscription never terminated"
				}
			})
		}

		close(start)
		waitBounded(t, &wg, "shared subscribers")

		close(errs)

		for msg := range errs {
			t.Fatal(msg)
		}

		if atomic.LoadInt64(&failed) != 0 {
			t.Fatalf("rohttpclient: %d of %d subscriptions failed", failed, n)
		}
	})
}

// FuzzHTTPRequestSubscribeContextCancel cancels the context given to
// SubscribeWithContext while the server is still working. The request should be
// aborted, or the observer notified, instead of the context being ignored.
func FuzzHTTPRequestSubscribeContextCancel(f *testing.F) {
	f.Skip("race: http-subscribe-ctx-ignored; remove when fixed")

	addSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		// The handler blocks until the client goes away or the test ends; a
		// request whose context was cancelled must show up as Done within fuzzCancelWindow.
		aborted := make(chan struct{})
		release := make(chan struct{})
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			select {
			case <-r.Context().Done():
				close(aborted)
			case <-release:
			}
		}))
		defer srv.Close()
		defer close(release)

		req, err := http.NewRequest(http.MethodGet, srv.URL, nil)
		if err != nil {
			t.Fatalf("rohttpclient: new request: %v", err)
		}

		ctx, cancel := context.WithCancel(context.Background())

		terminated := make(chan struct{})
		var once sync.Once
		finish := func() { once.Do(func() { close(terminated) }) }

		sub := HTTPRequest(req, srv.Client()).SubscribeWithContext(ctx, ro.NewObserver(
			func(res *http.Response) { _ = res.Body.Close() },
			func(error) { finish() },
			func() { finish() },
		))
		defer sub.Unsubscribe()

		// Sync mode cancels right away, async mode lets the request reach the server.
		if mask&fuzzModeAsync != 0 {
			time.Sleep(fuzzDelay(seed, mask))
		}

		cancel()

		select {
		case <-aborted:
		case <-terminated:
		case <-time.After(fuzzCancelWindow):
			t.Fatalf("rohttpclient: subscriber context cancelled but request neither aborted nor terminated")
		}
	})
}

// trackedBody records Close so the test can detect an undelivered response.
type trackedBody struct {
	io.ReadCloser
	closed *int64
}

func (b trackedBody) Close() error {
	atomic.AddInt64(b.closed, 1)
	return b.ReadCloser.Close()
}

// gateTransport returns the real response only after the subscription context
// is cancelled in async mode, reproducing the window where Unsubscribe lands
// between client.Do returning and destination.Next.
type gateTransport struct {
	inner     http.RoundTripper
	closed    *int64
	handedOut *int64
	returned  chan struct{}
	arrived   chan struct{}
	gate      bool
}

func (g gateTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	res, err := g.inner.RoundTrip(r)
	if err != nil {
		close(g.returned)
		return nil, err
	}

	if g.gate {
		// The response exists; hold it until the subscriber has unsubscribed.
		close(g.arrived)
		<-r.Context().Done()
	}

	res.Body = trackedBody{ReadCloser: res.Body, closed: g.closed}
	atomic.AddInt64(g.handedOut, 1)
	close(g.returned)

	return res, nil
}

// FuzzHTTPRequestUnsubscribeLeaksBody unsubscribes while a response is in
// flight. A response that is neither delivered nor closed leaks its connection.
func FuzzHTTPRequestUnsubscribeLeaksBody(f *testing.F) {
	f.Skip("race: http-unsubscribe-body-leak; remove when fixed")

	addSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		srv := newFuzzServer(fuzzDelay(seed, mask), "payload")
		defer srv.Close()

		var closed, delivered, handedOut int64

		returned := make(chan struct{})
		arrived := make(chan struct{})
		client := &http.Client{Transport: gateTransport{
			inner:     http.DefaultTransport,
			closed:    &closed,
			handedOut: &handedOut,
			returned:  returned,
			arrived:   arrived,
			gate:      mask&fuzzModeAsync != 0,
		}}

		req, err := http.NewRequest(http.MethodGet, srv.URL, nil)
		if err != nil {
			t.Fatalf("rohttpclient: new request: %v", err)
		}

		sub := HTTPRequest(req, client).Subscribe(ro.NewObserver(
			func(res *http.Response) {
				atomic.AddInt64(&delivered, 1)
				_ = res.Body.Close()
			},
			func(error) {},
			func() {},
		))

		// Async mode unsubscribes exactly once the response is held by the
		// transport; sync mode unsubscribes immediately and lets the scheduler pick.
		if mask&fuzzModeAsync != 0 {
			select {
			case <-arrived:
			case <-returned:
			case <-time.After(fuzzWait):
				t.Fatalf("rohttpclient: response never arrived")
			}
		}

		sub.Unsubscribe()

		select {
		case <-returned:
		case <-time.After(fuzzWait):
			t.Fatalf("rohttpclient: round trip never returned")
		}

		// Give the goroutine time to hand the response over or drop it.
		time.Sleep(20 * time.Millisecond)

		// A failed round trip hands out nothing, so only compare when a response
		// really left the transport.
		if atomic.LoadInt64(&handedOut) == 1 && atomic.LoadInt64(&delivered) == 0 && atomic.LoadInt64(&closed) == 0 {
			t.Fatalf("rohttpclient: response dropped after Unsubscribe without closing its body (delivered=0 closed=0)")
		}
	})
}

// FuzzHTTPRequestBodyReadAfterComplete reads the body once the stream has
// completed: callers are told to close the body themselves, so it must remain
// readable after Complete even though teardown cancels the request context.
func FuzzHTTPRequestBodyReadAfterComplete(f *testing.F) {
	f.Skip("race: http-body-ctx-canceled; remove when fixed")

	addSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		const payloadSize = 256 * 1024 // larger than the server write buffer, so the body streams

		delay := fuzzDelay(seed, mask)
		payload := make([]byte, payloadSize)

		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write(payload[:payloadSize/2])
			if fl, ok := w.(http.Flusher); ok {
				fl.Flush()
			}

			// Async mode stalls mid-body so the read happens after Complete.
			if delay > 0 {
				time.Sleep(delay)
			}

			_, _ = w.Write(payload[payloadSize/2:])
		}))
		defer srv.Close()

		req, err := http.NewRequest(http.MethodGet, srv.URL, nil)
		if err != nil {
			t.Fatalf("rohttpclient: new request: %v", err)
		}

		resCh := make(chan *http.Response, 1)
		completed := make(chan struct{})
		errCh := make(chan error, 1)

		sub := HTTPRequest(req, srv.Client()).Subscribe(ro.NewObserver(
			func(res *http.Response) { resCh <- res },
			func(err error) { errCh <- err },
			func() { close(completed) },
		))
		defer sub.Unsubscribe()

		var res *http.Response

		select {
		case res = <-resCh:
		case err := <-errCh:
			t.Fatalf("rohttpclient: request failed: %v", err)
		case <-time.After(fuzzWait):
			t.Fatalf("rohttpclient: no response within %s", fuzzWait)
		}

		defer res.Body.Close()

		select {
		case <-completed:
		case <-time.After(fuzzWait):
			t.Fatalf("rohttpclient: stream never completed")
		}

		readDone := make(chan error, 1)
		go func() {
			n, err := io.Copy(io.Discard, res.Body)
			if err == nil && n != payloadSize {
				err = fmt.Errorf("short body: %d of %d bytes", n, payloadSize)
			}

			readDone <- err
		}()

		select {
		case err := <-readDone:
			if err != nil {
				t.Fatalf("rohttpclient: reading body after Complete: %v", err)
			}
		case <-time.After(fuzzWait):
			t.Fatalf("rohttpclient: body read hung")
		}
	})
}
