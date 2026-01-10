# Time Plugin

The time plugin provides operators for manipulating dates/time object in reactive streams.

## Installation

```bash
go get github.com/samber/ro/plugins/time
```

## Operators

### Add

Add a duration to a date

```go
import (
    "github.com/samber/ro"
    rotime "github.com/samber/ro/plugins/time"
)

observable := ro.Pipe1(
    ro.Just(
        time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC),
        time.Date(0, time.January, 1, 23, 59, 59, 0, time.UTC),
    ),
    rotime.Add(2 * time.Hour),
)

// Output:
// Next: time.Date(2026, time.January, 7, 16, 30, 0, 0, time.UTC)
// Next: time.Date(0, time.January, 2, 1, 59, 59, 0, time.UTC)
// Completed
```

### AddDate

Add a duration defined by years, months, days to a date.

```go
import (
    "github.com/samber/ro"
    rotime "github.com/samber/ro/plugins/time"
)

observable := ro.Pipe1(
    ro.Just(
        time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC),
    ),
    rotime.AddDate(0, 1, 0),
)

// Output:
// Next: time.Date(2026, time.February, 7, 14, 30, 0, 0, time.UTC)
// Completed
```

### Format

Transform an observable time.Time into a string, formatted according to the provided layout.

```go
import (
    "github.com/samber/ro"
    rotime "github.com/samber/ro/plugins/time"
)

observable := ro.Pipe1(
    ro.Just(
        time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC),
    ),
    rotime.Format("2006-01-02 15:04:05"),
)

// Output:
// Next: "2026-01-07 14:30:00"
// Completed
```

### In

Transform an observable time.Time into a string, formatted according to the provided layout.

```go
import (
    "github.com/samber/ro"
    rotime "github.com/samber/ro/plugins/time"
)

observable := ro.Pipe1(
    ro.Just(
       time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC),
    ),
    rotime.In(time.LoadLocation("Europe/Paris")),
)

// Output:
// Next: time.Date(2026, time.January, 7, 16, 30, 0, 0, time.CET),
// Completed
```

### Parse

Transform an observable string into an observable of time.Time.

```go
import (
    "github.com/samber/ro"
    rotime "github.com/samber/ro/plugins/time"
)

observable := ro.Pipe1(
    ro.Just(
      "2026-01-07 14:30:00",
    ),
    rotime.Parse("2006-01-02 15:04:05"),
)

// Output:
// Next: time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC)
// Completed
```

### ParseInLocation

Transform an observable string into an observable of time.Time, using location.

```go
import (
    "github.com/samber/ro"
    rotime "github.com/samber/ro/plugins/time"
)

observable := ro.Pipe1(
    ro.Just(
      "2026-01-07 14:30:00",
    ),
    rotime.ParseInLocation("2006-01-02 15:04:05", time.UTC),
)

// Output:
// Next: time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC)
// Completed
```

### StartOfDay

Truncates the time to the beginning of its day in the local time zone

```go
import (
    "github.com/samber/ro"
    rotime "github.com/samber/ro/plugins/time"
)

observable := ro.Pipe1(
    ro.Just(
      time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC),
    ),
    rotime.StartOfDay(),
)

// Output:
// Next: time.Date(2026, time.January, 7, 0, 0, 0, 0, time.UTC)
// Completed
```

## Performance Considerations
- The time plugin uses Go's standard `time` package for operations