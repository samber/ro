---
name: NewPSINotifier
slug: newpsinotifier
sourceRef: plugins/samber/psi/source.go#L26
type: plugin
category: samber-psi
signatures:
  - "func NewPSINotifier(interval time.Duration)"
playUrl: ""
variantHelpers:
  - plugin#samber-psi#newpsinotifier
similarHelpers: []
position: 0
---

Creates PSI (Pressure Stall Information) notifier.

```go
import (
    "time"

    "github.com/samber/ro"
    ropsi "github.com/samber/ro/plugins/samber/psi"
    "github.com/shirou/gopsutil/v3/psinotifier"
)

obs := ropsi.NewPSINotifier(1 * time.Second)

sub := obs.Subscribe(ro.PrintObserver[psinotifier.PSIStatsResource]())
defer sub.Unsubscribe()

// Next: {Resource: "cpu", Avg10: 0.1, Avg60: 0.05, Avg300: 0.01, Total: 1000}
// Next: {Resource: "memory", Avg10: 0.2, Avg60: 0.1, Avg300: 0.05, Total: 2000}
// ...
```