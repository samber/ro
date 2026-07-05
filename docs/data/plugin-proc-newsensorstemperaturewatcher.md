---
name: NewSensorsTemperatureWatcher
slug: newsensorstemperaturewatcher
sourceRef: plugins/proc/source.go#L423
type: plugin
category: proc
signatures:
  - "func NewSensorsTemperatureWatcher(interval time.Duration, perNIC bool)"
playUrl: ""
variantHelpers:
  - plugin#proc#newsensorstemperaturewatcher
similarHelpers:
  - plugin#proc#newcpuinfowatcher
  - plugin#proc#newvirtualmemorywatcher
position: 41
---

Watches sensor temperature statistics.

```go
import (
    "time"

    "github.com/samber/ro"
    roproc "github.com/samber/ro/plugins/proc"
    "github.com/shirou/gopsutil/v4/sensors"
)

obs := roproc.NewSensorsTemperatureWatcher(2 * time.Second, false)

sub := obs.Subscribe(ro.PrintObserver[sensors.TemperatureStat]())
defer sub.Unsubscribe()

// Next: {SensorKey: coretemp-isa-0000, Temperature: 45.0, SensorHigh: 80.0, SensorCrit: 100.0}
// Next: {SensorKey: acpi-thermal-0, Temperature: 40.0, SensorHigh: 85.0, SensorCrit: 105.0}
// ...
```