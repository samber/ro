---
name: NewNetConntrackWatcher
slug: newnetconntrackwatcher
sourceRef: plugins/proc/source.go#L342
type: plugin
category: proc
signatures:
  - "func NewNetConntrackWatcher(interval time.Duration, perCPU bool)"
playUrl: ""
variantHelpers:
  - plugin#proc#newnetconntrackwatcher
similarHelpers:
  - plugin#proc#newnetconnectionswatcher
  - plugin#proc#newnetfiltercounterswatcher
position: 39
---

Watches netfilter conntrack statistics.

```go
import (
    "time"

    "github.com/samber/ro"
    roproc "github.com/samber/ro/plugins/proc"
    "github.com/shirou/gopsutil/v4/net"
)

obs := roproc.NewNetConntrackWatcher(2 * time.Second, false)

sub := obs.Subscribe(ro.PrintObserver[net.ConntrackStat]())
defer sub.Unsubscribe()

// Next: {Entries: 1000, Searched: 2000, Found: 1500, New: 50, ...}
// Next: {Entries: 1050, Searched: 2100, Found: 1575, New: 55, ...}
// ...
```