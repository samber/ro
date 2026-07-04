---
name: In
slug: in
sourceRef: plugins/time/operator_in_time_zone.go#L35
type: plugin
category: time
signatures:
  - "func In(loc *time.Location)"
playUrl: ""
variantHelpers:
  - plugin#time#in
similarHelpers:
  - plugin#time#startofday
  - plugin#time#add
position: 30
---

Converts each time.Time emitted by the source Observable to the given time zone location.

```go
import (
    "time"

    "github.com/samber/ro"
    rotime "github.com/samber/ro/plugins/time"
)

loc, _ := time.LoadLocation("America/New_York")

obs := ro.Pipe[time.Time, time.Time](
    ro.Just(time.Date(2024, 1, 15, 12, 0, 0, 0, time.UTC)),
    rotime.In(loc),
)

sub := obs.Subscribe(ro.PrintObserver[time.Time]())
defer sub.Unsubscribe()

// Next: 2024-01-15 07:00:00 -0500 EST
// Completed
```
