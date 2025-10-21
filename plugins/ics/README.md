# iCal/ICS Plugin

The ICS plugin provides operators for reading and processing iCalendar (ICS) files and URLs. It uses the `github.com/arran4/golang-ical` library to parse calendar data and emit individual events as reactive streams.

## Installation

```bash
go get github.com/samber/ro/plugins/ics
```

## Operators

### NewICSFileReader

Creates an observable that reads events from one or more ICS files.

```go
import (
    "github.com/samber/ro"
    roics "github.com/samber/ro/plugins/ics"
    ics "github.com/arran4/golang-ical"
)

observable := roics.NewICSFileReader(
    "calendar1.ics",
    "calendar2.ics",
    "calendar3.ics",
)

subscription := observable.Subscribe(ro.PrintObserver[*ics.VEvent]())
defer subscription.Unsubscribe()

// Output:
// Next: &{...} // VEvent objects
// Next: &{...}
// Next: &{...}
// Completed
```

### NewICSURLReader

Creates an observable that reads events from one or more ICS URLs.

```go
observable := roics.NewICSURLReader(
    "https://raw.githubusercontent.com/samber/ro/refs/heads/main/plugins/ics/testdata/fr-public-holidays-a.ics",
    "https://raw.githubusercontent.com/samber/ro/refs/heads/main/plugins/ics/testdata/fr-public-holidays-b.ics",
    "https://raw.githubusercontent.com/samber/ro/refs/heads/main/plugins/ics/testdata/fr-public-holidays-c.ics",
)

subscription := observable.Subscribe(ro.PrintObserver[*ics.VEvent]())
defer subscription.Unsubscribe()
```

## Working with VEvent Objects

The plugin emits `*ics.VEvent` objects that contain all the event information:

```go
import (
    "fmt"
    "time"
    "github.com/samber/ro"
    roics "github.com/samber/ro/plugins/ics"
    ics "github.com/arran4/golang-ical"
)

observable := roics.NewICSFileReader("calendar.ics")

subscription := observable.Subscribe(
    ro.NewObserver(
        func(event *ics.VEvent) {
            // Access event properties
            summary := event.GetProperty(ics.ComponentPropertySummary)
            if summary != nil {
                fmt.Printf("Event: %s\n", summary.Value)
            }
            
            start := event.GetProperty(ics.ComponentPropertyDtStart)
            if start != nil {
                fmt.Printf("Start: %s\n", start.Value)
            }
            
            end := event.GetProperty(ics.ComponentPropertyDtEnd)
            if end != nil {
                fmt.Printf("End: %s\n", end.Value)
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
defer subscription.Unsubscribe()
```

## Event Processing Examples

### Count Events

```go
import (
    "github.com/samber/ro"
    roics "github.com/samber/ro/plugins/ics"
    ics "github.com/arran4/golang-ical"
)

observable := ro.Pipe1(
    roics.NewICSFileReader("calendar.ics"),
    ro.Count[*ics.VEvent](),
)

subscription := observable.Subscribe(ro.PrintObserver[int64]())
defer subscription.Unsubscribe()

// Output:
// Next: 183
// Completed
```

### Filter Events by Date

```go
import (
    "time"
    "github.com/samber/ro"
    roics "github.com/samber/ro/plugins/ics"
    ics "github.com/arran4/golang-ical"
)

observable := ro.Pipe2(
    roics.NewICSFileReader("calendar.ics"),
    ro.Filter(func(event *ics.VEvent) bool {
        start := event.GetProperty(ics.ComponentPropertyDtStart)
        if start == nil {
            return false
        }
        
        eventTime, err := time.Parse("20060102T150405Z", start.Value)
        if err != nil {
            return false
        }
        
        // Filter events from 2024 onwards
        return eventTime.Year() >= 2024
    }),
)

subscription := observable.Subscribe(ro.PrintObserver[*ics.VEvent]())
defer subscription.Unsubscribe()
```

### Extract Event Summaries

```go
import (
    "github.com/samber/ro"
    roics "github.com/samber/ro/plugins/ics"
    ics "github.com/arran4/golang-ical"
)

observable := ro.Pipe2(
    roics.NewICSFileReader("calendar.ics"),
    ro.Map(func(event *ics.VEvent) string {
        summary := event.GetProperty(ics.ComponentPropertySummary)
        if summary != nil {
            return summary.Value
        }
        return "No title"
    }),
)

subscription := observable.Subscribe(ro.PrintObserver[string]())
defer subscription.Unsubscribe()

// Output:
// Next: Vacances de la Toussaint
// Next: Vacances de Noël
// Next: Vacances d'Hiver
// Completed
```

## Multiple Calendar Sources

You can process multiple calendar files or URLs in a single stream:

```go
// Process multiple local files
observable := roics.NewICSFileReader(
    "work-calendar.ics",
    "personal-calendar.ics",
    "holidays.ics",
)

// Process multiple URLs
observable := roics.NewICSURLReader(
    "https://calendar.google.com/calendar/ical/work%40company.com/public/basic.ics",
    "https://calendar.google.com/calendar/ical/personal%40gmail.com/public/basic.ics",
)

subscription := observable.Subscribe(ro.PrintObserver[*ics.VEvent]())
defer subscription.Unsubscribe()
```

## Error Handling

Both `ICSFileReader` and `ICSURLReader` handle errors gracefully:

### File Reading Errors

```go
observable := roics.NewICSFileReader("nonexistent.ics")

subscription := observable.Subscribe(
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
defer subscription.Unsubscribe()
```

### URL Reading Errors

```go
observable := roics.NewICSURLReader("https://invalid-url.com/calendar.ics")

subscription := observable.Subscribe(
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
defer subscription.Unsubscribe()
```

## Real-world Example

Here's a practical example that processes calendar events and extracts useful information:

```go
import (
    "fmt"
    "strings"
    "time"
    "github.com/samber/ro"
    roics "github.com/samber/ro/plugins/ics"
    ics "github.com/arran4/golang-ical"
)

type EventInfo struct {
    Title       string
    StartTime   string
    EndTime     string
    Description string
}

// Process calendar events and extract structured information
pipeline := ro.Pipe3(
    // Read from multiple calendar sources
    roics.NewICSFileReader(
        "work-calendar.ics",
        "personal-calendar.ics",
    ),
    // Filter events for the current year
    ro.Filter(func(event *ics.VEvent) bool {
        start := event.GetProperty(ics.ComponentPropertyDtStart)
        if start == nil {
            return false
        }
        
        eventTime, err := time.Parse("20060102T150405Z", start.Value)
        if err != nil {
            return false
        }
        
        return eventTime.Year() == time.Now().Year()
    }),
    // Transform to structured data
    ro.Map(func(event *ics.VEvent) EventInfo {
        summary := event.GetProperty(ics.ComponentPropertySummary)
        start := event.GetProperty(ics.ComponentPropertyDtStart)
        end := event.GetProperty(ics.ComponentPropertyDtEnd)
        description := event.GetProperty(ics.ComponentPropertyDescription)
        
        title := "No title"
        if summary != nil {
            title = summary.Value
        }
        
        startTime := "Unknown"
        if start != nil {
            startTime = start.Value
        }
        
        endTime := "Unknown"
        if end != nil {
            endTime = end.Value
        }
        
        desc := ""
        if description != nil {
            desc = description.Value
        }
        
        return EventInfo{
            Title:       title,
            StartTime:   startTime,
            EndTime:     endTime,
            Description: desc,
        }
    }),
)

subscription := pipeline.Subscribe(
    ro.NewObserver(
        func(event EventInfo) {
            fmt.Printf("Event: %s\n", event.Title)
            fmt.Printf("  Start: %s\n", event.StartTime)
            fmt.Printf("  End: %s\n", event.EndTime)
            if event.Description != "" {
                fmt.Printf("  Description: %s\n", event.Description)
            }
            fmt.Println()
        },
        func(err error) {
            fmt.Printf("Error: %v\n", err)
        },
        func() {
            fmt.Println("Processing completed")
        },
    ),
)
defer subscription.Unsubscribe()
```

## Performance Considerations

- The plugin uses the `github.com/arran4/golang-ical` library for parsing ICS files
- File reading is done sequentially for each provided path
- URL reading fetches calendars in sequence
- Large calendar files are processed efficiently with streaming
- The plugin handles context cancellation properly
- Consider the size and number of calendar files when processing multiple sources
- Network requests for URLs are made sequentially to avoid overwhelming servers

## Supported ICS Features

The plugin supports standard iCalendar features through the underlying library:

- VEVENT components (calendar events)
- VTIMEZONE components (timezone information)
- Recurring events (RRULE)
- Event properties (SUMMARY, DTSTART, DTEND, DESCRIPTION, etc.)
- Multiple calendar files and URLs
- Standard ICS format parsing

## Dependencies

The plugin depends on:
- `github.com/arran4/golang-ical` - iCalendar parsing library
- `github.com/samber/ro` - Reactive streams framework

Make sure to include these dependencies in your `go.mod` file when using the plugin.
