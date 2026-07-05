---
name: StartOfDay
slug: startofday
sourceRef: plugins/time/operator_start_of_day.go#L34
type: plugin
category: time
signatures:
  - "func StartOfDay()"
playUrl: https://go.dev/play/p/Rp_5REv3dMK
variantHelpers:
  - plugin#time#startofday
similarHelpers:
  - plugin#time#in
  - plugin#time#add
position: 40
---

Truncates each time.Time emitted by the source Observable to midnight (00:00:00) of the same day and location.

```go
import (
    "time"

    "github.com/samber/ro"
    rotime "github.com/samber/ro/plugins/time"
)

obs := ro.Pipe[time.Time, time.Time](
    ro.Just(
        time.Date(2024, 1, 15, 14, 30, 45, 0, time.UTC),
        time.Date(2024, 6, 20, 8, 15, 0, 0, time.UTC),
    ),
    rotime.StartOfDay(),
)

sub := obs.Subscribe(ro.PrintObserver[time.Time]())
defer sub.Unsubscribe()

// Next: 2024-01-15 00:00:00 +0000 UTC
// Next: 2024-06-20 00:00:00 +0000 UTC
// Completed
```
