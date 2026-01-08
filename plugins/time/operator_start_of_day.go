// obs := ro.Pipe[time.Time, time.Time](
//
//	ro.Just(time.Now()),
//	rotime.StartOfDay(),
//
// )
// // Next: "2026-01-06 00:00:00"
package rotime

import (
	"time"

	"github.com/samber/ro"
)

// StartOfDay truncates the time to the beginning of its day in the local time zone.
func StartOfDay() func(ro.Observable[time.Time]) ro.Observable[time.Time] {
	return ro.Map(
		func(value time.Time) time.Time {
			year, month, day := value.Date()
			return time.Date(year, month, day, 0, 0, 0, 0, value.Location())
		},
	)
}
