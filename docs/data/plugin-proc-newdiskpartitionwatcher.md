---
name: NewDiskPartitionWatcher
slug: newdiskpartitionwatcher
sourceRef: plugins/proc/source.go#L186
type: plugin
category: proc
signatures:
  - "func NewDiskPartitionWatcher(interval time.Duration)"
playUrl: ""
variantHelpers:
  - plugin#proc#newdiskpartitionwatcher
similarHelpers:
  - plugin#proc#newdiskusagewatcher
  - plugin#proc#newdiskiocounterswatcher
position: 35
---

Watches disk partition statistics.

```go
import (
    "time"

    "github.com/samber/ro"
    roproc "github.com/samber/ro/plugins/proc"
    "github.com/shirou/gopsutil/v4/disk"
)

obs := roproc.NewDiskPartitionWatcher(5 * time.Second)

sub := obs.Subscribe(ro.PrintObserver[disk.PartitionStat]())
defer sub.Unsubscribe()

// Next: {Device: /dev/sda1, Mountpoint: /, Fstype: ext4, Opts: rw,relatime}
// Next: {Device: /dev/sda2, Mountpoint: /home, Fstype: ext4, Opts: rw,relatime}
// ...
```