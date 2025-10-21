package ro

import (
    "context"
    "io/ioutil"
    "net/http"
    "os"
    "time"
)

// WatchFile creates an Observable that polls a file path at the given interval
// and emits the file contents as string when it changes. It emits on subscribe
// immediately the current contents.
func WatchFile(path string, interval time.Duration) Observable[string] {
    return NewObservableWithContext(func(ctx context.Context, destination Observer[string]) Teardown {
        var last []byte

        // send initial value if file exists
        if b, err := ioutil.ReadFile(path); err == nil {
            last = b
            destination.NextWithContext(ctx, string(b))
        }

        ticker := time.NewTicker(interval)
        done := make(chan struct{})

        go recoverUnhandledError(func() {
            defer destination.CompleteWithContext(ctx)
            for {
                select {
                case <-done:
                    return
                case <-ctx.Done():
                    return
                case <-ticker.C:
                    b, err := ioutil.ReadFile(path)
                    if err != nil {
                        // If file not found, skip
                        if !os.IsNotExist(err) {
                            destination.ErrorWithContext(ctx, err)
                            return
                        }
                        continue
                    }

                    if len(b) != len(last) || string(b) != string(last) {
                        last = b
                        destination.NextWithContext(ctx, string(b))
                    }
                }
            }
        })

        return func() {
            ticker.Stop()
            close(done)
        }
    })
}

// WatchURL creates an Observable that polls the given URL at interval and
// emits the response body as string when it changes. It emits initial value
// immediately on subscribe.
func WatchURL(url string, interval time.Duration) Observable[string] {
    return NewObservableWithContext(func(ctx context.Context, destination Observer[string]) Teardown {
        client := &http.Client{Timeout: 10 * time.Second}
        var last []byte

        // initial
        if resp, err := client.Get(url); err == nil {
            b, _ := ioutil.ReadAll(resp.Body)
            resp.Body.Close()
            last = b
            destination.NextWithContext(ctx, string(b))
        }

        ticker := time.NewTicker(interval)
        done := make(chan struct{})

        go recoverUnhandledError(func() {
            defer destination.CompleteWithContext(ctx)
            for {
                select {
                case <-done:
                    return
                case <-ctx.Done():
                    return
                case <-ticker.C:
                    resp, err := client.Get(url)
                    if err != nil {
                        destination.ErrorWithContext(ctx, err)
                        return
                    }
                    b, _ := ioutil.ReadAll(resp.Body)
                    resp.Body.Close()

                    if len(b) != len(last) || string(b) != string(last) {
                        last = b
                        destination.NextWithContext(ctx, string(b))
                    }
                }
            }
        })

        return func() {
            ticker.Stop()
            close(done)
        }
    })
}
