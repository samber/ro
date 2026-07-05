---
name: NewLoadMiscWatcher
slug: newloadmiscwatcher
sourceRef: plugins/proc/source.go#L290
type: plugin
category: proc
signatures:
  - "func NewLoadMiscWatcher(interval time.Duration)"
playUrl: ""
variantHelpers:
  - plugin#proc#newloadmiscwatcher
similarHelpers:
  - plugin#proc#newloadaveragewatcher
  - plugin#proc#newcpuinfowatcher
position: 37
---

Watches miscellaneous load statistics.

```go
import (
    "time"

    "github.com/samber/ro"
    roproc "github.com/samber/ro/plugins/proc"
    "github.com/shirou/gopsutil/v4/load"
)

obs := roproc.NewLoadMiscWatcher(2 * time.Second)

sub := obs.Subscribe(ro.PrintObserver[*load.MiscStat]())
defer sub.Unsubscribe()

// Next: &{ProcsTotal: 150, ProcsRunning: 2, ProcsBlocked: 1, CtxSwitches: 1000000, ...}
// Next: &{ProcsTotal: 152, ProcsRunning: 3, ProcsBlocked: 1, CtxSwitches: 1001000, ...}
// ...
```