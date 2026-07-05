---
name: NewSwapMemoryWatcher
slug: newswapmemorywatcher
sourceRef: plugins/proc/source.go#L57
type: plugin
category: proc
signatures:
  - "func NewSwapMemoryWatcher(interval time.Duration)"
playUrl: ""
variantHelpers:
  - plugin#proc#newswapmemorywatcher
similarHelpers:
  - plugin#proc#newvirtualmemorywatcher
  - plugin#proc#newswapdevicewatcher
position: 32
---

Watches swap memory statistics.

```go
import (
    "time"

    "github.com/samber/ro"
    roproc "github.com/samber/ro/plugins/proc"
    "github.com/shirou/gopsutil/v4/mem"
)

obs := roproc.NewSwapMemoryWatcher(2 * time.Second)

sub := obs.Subscribe(ro.PrintObserver[*mem.SwapMemoryStat]())
defer sub.Unsubscribe()

// Next: &{Total: 4294967296, Used: 2147483648, Free: 2147483648, ...}
// Next: &{Total: 4294967296, Used: 2155892256, Free: 2139075040, ...}
// ...
```