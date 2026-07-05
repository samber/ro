---
name: NewLoadAverageWatcher
slug: newloadaveragewatcher
sourceRef: plugins/proc/source.go#L265
type: plugin
category: proc
signatures:
  - "func NewLoadAverageWatcher(interval time.Duration)"
playUrl: ""
variantHelpers:
  - plugin#proc#newloadaveragewatcher
similarHelpers:
  - plugin#proc#newvirtualmemorywatcher
  - plugin#proc#newcpuinfowatcher
  - plugin#proc#newhostinfowatcher
position: 60
---

Emits system load average statistics at regular intervals.

```go
import (
    "time"
    "github.com/samber/ro"
    roproc "github.com/samber/ro/plugins/proc"
)

obs := roproc.NewLoadAverageWatcher(2 * time.Second)

sub := obs.Subscribe(ro.PrintObserver[*load.AvgStat]())
defer sub.Unsubscribe()

// Next: &{Load1: 0.75 Load5: 0.82 Load15: 0.90}
// Next: &{Load1: 0.80 Load5: 0.83 Load15: 0.91}
// Next: &{Load1: 0.78 Load5: 0.81 Load15: 0.89}
// ... (continues every 2 seconds)
```

Returns system load average values for 1, 5, and 15 minute intervals.