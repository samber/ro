---
name: Parse
slug: parse
sourceRef: plugins/time/operator_parse.go#L33
type: plugin
category: time
signatures:
  - "func Parse[T ~string](layout string)"
  - "func ParseInLocation[T ~string](layout string, loc *time.Location)"
playUrl: ""
variantHelpers:
  - plugin#time#parse
  - plugin#time#parseinlocation
similarHelpers:
  - plugin#time#format
position: 0
---

Parses string values emitted by the source Observable into time.Time using the given layout.

```go
import (
    "time"

    "github.com/samber/ro"
    rotime "github.com/samber/ro/plugins/time"
)

obs := ro.Pipe[string, time.Time](
    ro.Just("2024-01-15", "2024-06-20"),
    rotime.Parse[string]("2006-01-02"),
)

sub := obs.Subscribe(ro.PrintObserver[time.Time]())
defer sub.Unsubscribe()

// Next: 2024-01-15 00:00:00 +0000 UTC
// Next: 2024-06-20 00:00:00 +0000 UTC
// Completed
```

### ParseInLocation

```go
loc, _ := time.LoadLocation("America/New_York")

obs := ro.Pipe[string, time.Time](
    ro.Just("2024-01-15 10:00:00"),
    rotime.ParseInLocation[string]("2006-01-02 15:04:05", loc),
)

sub := obs.Subscribe(ro.PrintObserver[time.Time]())
defer sub.Unsubscribe()

// Next: 2024-01-15 10:00:00 -0500 EST
// Completed
```
