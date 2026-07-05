---
name: NewHostUserWatcher
slug: newhostuserwatcher
sourceRef: plugins/proc/source.go#L238
type: plugin
category: proc
signatures:
  - "func NewHostUserWatcher(interval time.Duration)"
playUrl: ""
variantHelpers:
  - plugin#proc#newhostuserwatcher
similarHelpers:
  - plugin#proc#newhostinfowatcher
  - plugin#proc#newloadaveragewatcher
position: 36
---

Watches host user statistics.

```go
import (
    "time"

    "github.com/samber/ro"
    roproc "github.com/samber/ro/plugins/proc"
    "github.com/shirou/gopsutil/v4/host"
)

obs := roproc.NewHostUserWatcher(10 * time.Second)

sub := obs.Subscribe(ro.PrintObserver[host.UserStat]())
defer sub.Unsubscribe()

// Next: {User: root, Terminal: /dev/pts/0, Host: localhost, Started: 1640995200}
// Next: {User: samber, Terminal: /dev/pts/1, Host: localhost, Started: 1640995300}
// ...
```