---
name: NewDiskUsageWatcher
slug: newdiskusagewatcher
sourceRef: plugins/proc/source.go#L136
type: plugin
category: proc
signatures:
  - "func NewDiskUsageWatcher(interval time.Duration, mountpointOrDevicePath string)"
playUrl: ""
variantHelpers:
  - plugin#proc#newdiskusagewatcher
similarHelpers:
  - plugin#proc#newvirtualmemorywatcher
  - plugin#proc#newdiskiocounterswatcher
position: 70
---

Emits disk usage statistics for a specific mount point or device at regular intervals.

```go
import (
    "time"
    "github.com/samber/ro"
    roproc "github.com/samber/ro/plugins/proc"
)

obs := roproc.NewDiskUsageWatcher(5 * time.Second, "/")

sub := obs.Subscribe(ro.PrintObserver[*disk.UsageStat]())
defer sub.Unsubscribe()

// Next: &{Path: "/" Total: 500000000000 Free: 250000000000 Used: 250000000000 UsedPercent: 50.0 ...}
// Next: &{Path: "/" Total: 500000000000 Free: 249000000000 Used: 251000000000 UsedPercent: 50.2 ...}
// ... (continues every 5 seconds)
```

Returns disk usage information including total space, free space, used space, and usage percentage for the specified path.