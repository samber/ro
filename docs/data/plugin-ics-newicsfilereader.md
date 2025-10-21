---
name: NewICSFileReader
slug: newicsfilereader
sourceRef: source.go#L27
type: plugin
category: ics
signatures:
  - "func NewICSFileReader(paths ...string)"
playUrl:
variantHelpers:
  - plugin#ics#newicsfilereader
similarHelpers:
  - plugin#ics#newicsurlreader
position: 0
---

Reads events from one or more ICS files and emits them as a stream of VEvent objects.

```go
import (
    "github.com/samber/ro"
    roics "github.com/samber/ro/plugins/ics"
    ics "github.com/arran4/golang-ical"
)

obs := ro.Pipe1(
    roics.NewICSFileReader(
        "calendar1.ics",
        "calendar2.ics",
        "calendar3.ics",
    ),
    ro.Count[*ics.VEvent](),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
defer sub.Unsubscribe()

// Next: 183
// Completed
```

### Processing individual events

```go
obs := roics.NewICSFileReader("calendar.ics")

sub := obs.Subscribe(
    ro.NewObserver(
        func(event *ics.VEvent) {
            summary := event.GetProperty(ics.ComponentPropertySummary)
            if summary != nil {
                fmt.Printf("Event: %s\n", summary.Value)
            }
        },
        func(err error) {
            fmt.Printf("Error: %v\n", err)
        },
        func() {
            fmt.Println("Completed")
        },
    ),
)
defer sub.Unsubscribe()
```

### Error handling

```go
obs := roics.NewICSFileReader("nonexistent.ics")

sub := obs.Subscribe(
    ro.NewObserver(
        func(event *ics.VEvent) {
            // Handle successful event reading
        },
        func(err error) {
            // Handle file reading error
            // This could be due to:
            // - File not found
            // - Permission errors
            // - Invalid ICS format
            fmt.Printf("Error: %v\n", err)
        },
        func() {
            // Handle completion
        },
    ),
)
defer sub.Unsubscribe()
```

### Empty file list

```go
obs := roics.NewICSFileReader() // No files provided

sub := obs.Subscribe(ro.PrintObserver[*ics.VEvent]())
defer sub.Unsubscribe()

// Completed (empty observable)
```
