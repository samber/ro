package rotime

import (
	"time"

	"github.com/samber/ro"
)

// Format returns a function that transforms an observable of time.Time values
// into an observable of strings, formatted according to the provided layout.
// The layout must be a valid time format string (e.g., "2006-01-02 15:04:05").
func Format[T ~time.Time](format string) func(destination ro.Observable[T]) ro.Observable[string] {
	return ro.Map(
		func(value T) string {
			return value.Format(format)
		},
	)
}

//obs := ro.Pipe(
//    ro.Just(time.Now()),
//    rotime.Format("2006-01-02 15:04:05"),
//)
// obs now emits the current time as a formatted string, e.g., "2026-01-07 14:30:00"
