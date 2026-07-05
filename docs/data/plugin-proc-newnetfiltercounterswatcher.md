---
name: NewNetFilterCountersWatcher
slug: newnetfiltercounterswatcher
sourceRef: plugins/proc/source.go#L369
type: plugin
category: proc
signatures:
  - "func NewNetFilterCountersWatcher(interval time.Duration)"
playUrl: ""
variantHelpers:
  - plugin#proc#newnetfiltercounterswatcher
similarHelpers:
  - plugin#proc#newnetconntrackwatcher
  - plugin#proc#newnetiocounterswatcher
position: 40
---

Watches netfilter counter statistics.

```go
import (
    "time"

    "github.com/samber/ro"
    roproc "github.com/samber/ro/plugins/proc"
    "github.com/shirou/gopsutil/v4/net"
)

obs := roproc.NewNetFilterCountersWatcher(2 * time.Second)

sub := obs.Subscribe(ro.PrintObserver[net.FilterStat]())
defer sub.Unsubscribe()

// Next: {ConntrackCount: 1000, ConntrackMax: 65536}
// Next: {ConntrackCount: 1050, ConntrackMax: 65536}
// ...
```