---
name: NewCPUInfoWatcher
slug: newcpuinfowatcher
sourceRef: plugins/proc/source.go#L109
type: plugin
category: proc
signatures:
  - "func NewCPUInfoWatcher(interval time.Duration)"
playUrl: ""
variantHelpers:
  - plugin#proc#newcpuinfowatcher
similarHelpers:
  - plugin#proc#newvirtualmemorywatcher
  - plugin#proc#newloadaveragewatcher
position: 30
---

Watches CPU information.

```go
import (
    "time"

    "github.com/samber/ro"
    roproc "github.com/samber/ro/plugins/proc"
    "github.com/shirou/gopsutil/v3/cpu"
)

obs := roproc.NewCPUInfoWatcher(2 * time.Second)

sub := obs.Subscribe(ro.PrintObserver[cpu.InfoStat]())
defer sub.Unsubscribe()

// Next: {CPU: 0, VendorID: GenuineIntel, Family: 6, Model: 142 ...}
// Next: {CPU: 1, VendorID: GenuineIntel, Family: 6, Model: 142 ...}
// ...
```