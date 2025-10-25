---
name: NewScheduler
slug: newscheduler
sourceRef: plugins/cron/source.go#L38
type: plugin
category: cron
signatures:
  - "func NewScheduler(job gocron.JobDefinition)"
playUrl: https://go.dev/play/p/oWIqF0o0dZ8
variantHelpers:
  - plugin#cron#newscheduler
similarHelpers: []
position: 0
---

Creates an observable that emits notifications on scheduler ticks.

```go
import (
    "github.com/go-co-op/gocron"

    "github.com/samber/ro"
    rocron "github.com/samber/ro/plugins/cron"
)

job := gocron.Every(1).Second()
obs := rocron.NewScheduler(job)

sub := obs.Subscribe(ro.PrintObserver[ScheduleJob]())
defer sub.Unsubscribe()

// Next: ScheduleJob{...}
// Next: ScheduleJob{...}
// ...
```