---
name: NewICSURLReader
slug: newicsurlreader
sourceRef: source.go#L56
type: plugin
category: ics
signatures:
  - "func NewICSURLReader(urls ...string)"
playUrl:
variantHelpers:
  - plugin#ics#newicsurlreader
similarHelpers:
  - plugin#ics#newicsfilereader
position: 10
---

Reads events from one or more ICS URLs and emits them as a stream of VEvent objects.

```go
import (
    "github.com/samber/ro"
    roics "github.com/samber/ro/plugins/ics"
    ics "github.com/arran4/golang-ical"
)

obs := ro.Pipe1(
    roics.NewICSURLReader(
        "https://raw.githubusercontent.com/samber/ro/refs/heads/main/plugins/ics/testdata/fr-public-holidays-a.ics",
        "https://raw.githubusercontent.com/samber/ro/refs/heads/main/plugins/ics/testdata/fr-public-holidays-b.ics",
        "https://raw.githubusercontent.com/samber/ro/refs/heads/main/plugins/ics/testdata/fr-public-holidays-c.ics",
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
obs := roics.NewICSURLReader("https://calendar.google.com/calendar/ical/example%40gmail.com/public/basic.ics")

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
obs := roics.NewICSURLReader("https://invalid-url.com/calendar.ics")

sub := obs.Subscribe(
    ro.NewObserver(
        func(event *ics.VEvent) {
            // Handle successful event reading
        },
        func(err error) {
            // Handle URL reading error
            // This could be due to:
            // - Network errors
            // - Invalid URL
            // - Server errors
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

### Empty URL list

```go
obs := roics.NewICSURLReader() // No URLs provided

sub := obs.Subscribe(ro.PrintObserver[*ics.VEvent]())
defer sub.Unsubscribe()

// Completed (empty observable)
```
