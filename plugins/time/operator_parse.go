// obs := ro.Pipe[string, time.Time](
//
//	ro.Just("2026-01-06 14:30:00"),
//	rotime.Parse("2006-01-02 15:04:05"),
//
// )
// // Next: time.Time{...}
package rotime

import (
	"time"

	"github.com/samber/ro"
)

// Parse returns a function that transforms an observable of string-like values
// into an observable of time.Time. On parse error, the zero time is emitted.
func Parse[T ~string](layout string) func(ro.Observable[T]) ro.Observable[time.Time] {
	return ro.Map(
		func(value T) time.Time {
			t, err := time.Parse(layout, string(value))
			if err != nil {
				return time.Time{}
			}
			return t
		},
	)
}

func ParseInLocation[T ~string](layout string, loc *time.Location) func(ro.Observable[T]) ro.Observable[time.Time] {
	return ro.Map(
		func(value T) time.Time {
			t, err := time.ParseInLocation(layout, string(value), loc)
			if err != nil {
				return time.Time{}
			}
			return t
		},
	)
}
