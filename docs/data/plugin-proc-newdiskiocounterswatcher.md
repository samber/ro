---
name: NewDiskIOCountersWatcher
slug: newdiskiocounterswatcher
sourceRef: plugins/proc/source.go#L161
type: plugin
category: proc
signatures:
  - "func NewDiskIOCountersWatcher(interval time.Duration, names ...string)"
playUrl: ""
variantHelpers:
  - plugin#proc#newdiskiocounterswatcher
similarHelpers:
  - plugin#proc#newdiskusagewatcher
  - plugin#proc#newnetiocounterswatcher
position: 34
---

Watches disk I/O counters statistics.

```go
import (
    "time"

    "github.com/samber/ro"
    roproc "github.com/samber/ro/plugins/proc"
    "github.com/shirou/gopsutil/v4/disk"
)

obs := roproc.NewDiskIOCountersWatcher(2 * time.Second, "sda", "sdb")

sub := obs.Subscribe(ro.PrintObserver[map[string]disk.IOCountersStat]())
defer sub.Unsubscribe()

// Next: map[sda:{ReadCount: 1000, WriteCount: 500, ...} sdb:{ReadCount: 800, WriteCount: 300, ...}]
// Next: map[sda:{ReadCount: 1005, WriteCount: 505, ...} sdb:{ReadCount: 805, WriteCount: 305, ...}]
// ...
```