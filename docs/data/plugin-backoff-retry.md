---
name: Retry
slug: retry
sourceRef: plugins/backoff/operator.go#L53
type: plugin
category: backoff
signatures:
  - "func Retry[T any](strategy Strategy)"
  - "func RetryWithConfig[T any](config Config)"
playUrl:
variantHelpers:
  - plugin#backoff#retry
  - plugin#backoff#retrywithconfig
similarHelpers:
  - core#error-handling#catch
  - core#error-handling#retry
position: 0
---

Resubscribes to the source Observable when it errors, with an optional backoff delay between attempts.

`Retry` retries indefinitely using a given delay strategy. `RetryWithConfig` gives full control over the maximum number of retries, the delay strategy, and whether to reset the attempt counter after each successful item.

```go
import (
    "fmt"
    "time"

    "github.com/samber/ro"
    robackoff "github.com/samber/ro/plugins/backoff"
)

attempt := 0
src := ro.NewUnsafeObservable(func(dst ro.Observer[int]) ro.Teardown {
    attempt++
    if attempt < 3 {
        dst.Error(fmt.Errorf("transient error (attempt %d)", attempt))
        return nil
    }
    dst.Next(42)
    dst.Complete()
    return nil
})

values, err := ro.Collect(
    robackoff.RetryWithConfig[int](robackoff.Config{
        MaxRetries: 5,
        Strategy:   robackoff.Exponential(10*time.Millisecond, 2, 500*time.Millisecond),
    })(src),
)

fmt.Println(values, err)
// [42] <nil>
```

## Strategies

```go
// Fixed delay between every retry
robackoff.Constant(100 * time.Millisecond)

// Delay grows linearly: initial, initial+step, initial+2*step, ...
robackoff.Linear(50*time.Millisecond, 50*time.Millisecond, 1*time.Second)

// Delay grows exponentially: initial * factor^n, capped at maxDelay
robackoff.Exponential(100*time.Millisecond, 2.0, 30*time.Second)

// Adds full random jitter on top of any strategy (decorrelated jitter)
robackoff.FullJitter(robackoff.Exponential(100*time.Millisecond, 2.0, 30*time.Second))
```

