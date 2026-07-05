---
name: NewHostInfoWatcher
slug: newhostinfowatcher
sourceRef: plugins/proc/source.go#L213
type: plugin
category: proc
signatures:
  - "func NewHostInfoWatcher(interval time.Duration)"
playUrl: ""
variantHelpers:
  - plugin#proc#newhostinfowatcher
similarHelpers:
  - plugin#proc#newvirtualmemorywatcher
  - plugin#proc#newcpuinfowatcher
  - plugin#proc#newloadaveragewatcher
position: 80
---

Emits host system information at regular intervals.

```go
import (
    "time"
    "github.com/samber/ro"
    roproc "github.com/samber/ro/plugins/proc"
)

obs := roproc.NewHostInfoWatcher(10 * time.Second)

sub := obs.Subscribe(ro.PrintObserver[*host.InfoStat]())
defer sub.Unsubscribe()

// Next: &{Hostname: "my-server" Uptime: 86400 BootTime: 1640995200 Procs: 150 OS: "linux" ...}
// Next: &{Hostname: "my-server" Uptime: 86410 BootTime: 1640995200 Procs: 152 OS: "linux" ...}
// ... (continues every 10 seconds)
```

Returns comprehensive host information including hostname, uptime, boot time, process count, operating system details, platform, architecture, and other system metadata.