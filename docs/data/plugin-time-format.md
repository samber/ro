---
name: Format
slug: format
sourceRef: plugins/time/operator_format.go#L33
type: plugin
category: time
signatures:
  - "func Format(format string)"
playUrl: ""
variantHelpers:
  - plugin#time#format
similarHelpers:
  - plugin#time#parse
position: 10
---

Formats each time.Time emitted by the source Observable into a string using the given layout.

```go
import (
    "time"

    "github.com/samber/ro"
    rotime "github.com/samber/ro/plugins/time"
)

obs := ro.Pipe[time.Time, string](
    ro.Just(
        time.Date(2024, 1, 15, 12, 30, 0, 0, time.UTC),
        time.Date(2024, 6, 20, 8, 0, 0, 0, time.UTC),
    ),
    rotime.Format("2006-01-02"),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: 2024-01-15
// Next: 2024-06-20
// Completed
```
