# Backoff & Retry Plugin

The backoff plugin provides reactive retry operators with configurable backoff strategies (Constant, Linear, Exponential, FullJitter) for `ro.Observable` streams.

## Installation

```bash
go get github.com/samber/ro/plugins/backoff
```

## Operators

### Retry

Retries indefinitely when the source Observable emits an error, applying the specified delay strategy between attempts.

```go
import (
    "time"

    "github.com/samber/ro"
    robackoff "github.com/samber/ro/plugins/backoff"
)

observable := ro.Pipe1(
    source,
    robackoff.Retry[int](robackoff.Constant(500 * time.Millisecond)),
)
```

### RetryWithConfig

Provides fine-grained control over retry behavior:

- `MaxRetries`: Maximum number of retry attempts (`0` means unlimited).
- `Strategy`: A `Strategy` function calculating delay for each attempt.
- `ResetOnSuccess`: If `true`, resets the retry attempt counter back to 0 whenever the source emits a value.

```go
observable := ro.Pipe1(
    source,
    robackoff.RetryWithConfig[int](robackoff.Config{
        MaxRetries:     5,
        Strategy:       robackoff.Exponential(100*time.Millisecond, 2.0, 10*time.Second),
        ResetOnSuccess: true,
    }),
)
```

## Strategies

- **`Constant(delay)`** — Retries with a fixed delay every time.
- **`Linear(initial, step, maxDelay)`** — Linearly increasing delay: `initial + (attempt - 1) * step`, capped at `maxDelay`.
- **`Exponential(initial, factor, maxDelay)`** — Exponential backoff: `initial * factor^(attempt - 1)`, capped at `maxDelay`.
- **`FullJitter(strategy)`** — Wraps any strategy with full random jitter in `[0, delay]`.

