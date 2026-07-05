---
name: NewSwapDeviceWatcher
slug: newswapdevicewatcher
sourceRef: plugins/proc/source.go#L82
type: plugin
category: proc
signatures:
  - "func NewSwapDeviceWatcher(interval time.Duration)"
playUrl: ""
variantHelpers:
  - plugin#proc#newswapdevicewatcher
similarHelpers:
  - plugin#proc#newswapmemorywatcher
  - plugin#proc#newvirtualmemorywatcher
position: 33
---

Watches swap device statistics.

```go
import (
    "time"

    "github.com/samber/ro"
    roproc "github.com/samber/ro/plugins/proc"
    "github.com/shirou/gopsutil/v4/mem"
)

obs := roproc.NewSwapDeviceWatcher(2 * time.Second)

sub := obs.Subscribe(ro.PrintObserver[*mem.SwapDevice]())
defer sub.Unsubscribe()

// Next: &{Name: /dev/sda1, Used: 1073741824, Free: 3221225472, ...}
// Next: &{Name: /dev/sda2, Used: 2147483648, Free: 2147483648, ...}
// ...
```