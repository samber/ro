package rotime

import (
	"time"

	"github.com/samber/ro"
)

// obs := ro.Pipe[time.Time, time.Time](
// 	ro.Just(time.Now()),
// 	rotime.Add(2*time.Hour),
// )
// // Next: time.Now().Add(2 * time.Hour)

// // can do the same for sub

func Add(d time.Duration) func(destination ro.Observable[time.Time]) ro.Observable[time.Time] {
	return ro.Map(
		func(value time.Time) time.Time {
			return value.Add(d)
		},
	)
}

func AddDate(years int, months int, days int) func(destination ro.Observable[time.Time]) ro.Observable[time.Time] {
	return ro.Map(
		func(value time.Time) time.Time {
			return value.AddDate(years, months, days)
		},
	)
}
