---
name: Add
slug: add
sourceRef: plugins/time/operator_add.go#L34
type: plugin
category: time
signatures:
  - "func Add(d time.Duration)"
  - "func AddDate(years int, months int, days int)"
playUrl: https://go.dev/play/p/XWgGO-93YPK
variantHelpers:
  - plugin#time#add
  - plugin#time#adddate
similarHelpers:
  - plugin#time#startofday
  - plugin#time#in
position: 20
---

Adds a duration or date offset to each time.Time emitted by the source Observable.

```go
import (
    "time"

    "github.com/samber/ro"
    rotime "github.com/samber/ro/plugins/time"
)

obs := ro.Pipe[time.Time, time.Time](
    ro.Just(time.Date(2024, 1, 15, 12, 0, 0, 0, time.UTC)),
    rotime.Add(24 * time.Hour),
)

sub := obs.Subscribe(ro.PrintObserver[time.Time]())
defer sub.Unsubscribe()

// Next: 2024-01-16 12:00:00 +0000 UTC
// Completed
```

### AddDate

```go
obs := ro.Pipe[time.Time, time.Time](
    ro.Just(time.Date(2024, 1, 15, 0, 0, 0, 0, time.UTC)),
    rotime.AddDate(1, 2, 3),
)

sub := obs.Subscribe(ro.PrintObserver[time.Time]())
defer sub.Unsubscribe()

// Next: 2025-03-18 00:00:00 +0000 UTC
// Completed
```
