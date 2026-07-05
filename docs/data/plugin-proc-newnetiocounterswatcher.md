---
name: NewNetIOCountersWatcher
slug: newnetiocounterswatcher
sourceRef: plugins/proc/source.go#L396
type: plugin
category: proc
signatures:
  - "func NewNetIOCountersWatcher(interval time.Duration, perNIC bool)"
playUrl: ""
variantHelpers:
  - plugin#proc#newnetiocounterswatcher
similarHelpers:
  - plugin#proc#newvirtualmemorywatcher
  - plugin#proc#newdiskiocounterswatcher
  - plugin#proc#newnetconnectionswatcher
position: 90
---

Emits network IO counters statistics at regular intervals.

```go
import (
    "time"
    "github.com/samber/ro"
    roproc "github.com/samber/ro/plugins/proc"
)

obs := roproc.NewNetIOCountersWatcher(3 * time.Second, true)

sub := obs.Subscribe(ro.PrintObserver[net.IOCountersStat]())
defer sub.Unsubscribe()

// Next: &{Name: "eth0" BytesSent: 1024000 BytesRecv: 2048000 PacketsSent: 1000 PacketsRecv: 2000 ...}
// Next: &{Name: "eth0" BytesSent: 1025000 BytesRecv: 2050000 PacketsSent: 1005 PacketsRecv: 2005 ...}
// Next: &{Name: "lo" BytesSent: 5000 BytesRecv: 5000 PacketsSent: 50 PacketsRecv: 50 ...}
// ... (continues every 3 seconds)
```

Returns network interface statistics including bytes sent/received, packets sent/received, and other IO metrics. Set perNIC to true to get stats per network interface, or false to get aggregated stats.