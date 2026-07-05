---
name: NewVirtualMemoryWatcher
slug: newvirtualmemorywatcher
sourceRef: plugins/proc/source.go#L32
type: plugin
category: proc
signatures:
  - "func NewVirtualMemoryWatcher(interval time.Duration)"
playUrl: ""
variantHelpers:
  - plugin#proc#newvirtualmemorywatcher
similarHelpers:
  - plugin#proc#newswapmemorywatcher
  - plugin#proc#newcpuinfowatcher
position: 31
---

Watches virtual memory statistics.

```go
import (
    "time"

    "github.com/samber/ro"
    roproc "github.com/samber/ro/plugins/proc"
    "github.com/shirou/gopsutil/v4/mem"
)

obs := roproc.NewVirtualMemoryWatcher(2 * time.Second)

sub := obs.Subscribe(ro.PrintObserver[*mem.VirtualMemoryStat]())
defer sub.Unsubscribe()

// Next: &{Total: 17179869184, Available: 8589934592, Used: 8589934592, ...}
// Next: &{Total: 17179869184, Available: 85983232, Used: 8589934592, ...}
// ...
```