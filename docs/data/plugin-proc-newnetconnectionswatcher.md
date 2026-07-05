---
name: NewNetConnectionsWatcher
slug: newnetconnectionswatcher
sourceRef: plugins/proc/source.go#L315
type: plugin
category: proc
signatures:
  - "func NewNetConnectionsWatcher(interval time.Duration)"
playUrl: ""
variantHelpers:
  - plugin#proc#newnetconnectionswatcher
similarHelpers:
  - plugin#proc#newnetiocounterswatcher
  - plugin#proc#newnetconntrackwatcher
position: 38
---

Watches network connection statistics.

```go
import (
    "time"

    "github.com/samber/ro"
    roproc "github.com/samber/ro/plugins/proc"
    "github.com/shirou/gopsutil/v4/net"
)

obs := roproc.NewNetConnectionsWatcher(3 * time.Second)

sub := obs.Subscribe(ro.PrintObserver[net.ConnectionStat]())
defer sub.Unsubscribe()

// Next: {Fd: 3, Family: 2, Type: 1, Laddr: {IP: 127.0.0.1, Port: 8080}, ...}
// Next: {Fd: 4, Family: 2, Type: 1, Laddr: {IP: 192.168.1.100, Port: 443}, ...}
// ...
```