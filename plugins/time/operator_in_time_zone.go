package rotime

import (
	"time"

	"github.com/samber/ro"
)

func InTimeZone(loc *time.Location) func(destination ro.Observable[time.Time]) ro.Observable[time.Time] {
	return ro.Map(
		func(value time.Time) time.Time {
			return value.In(loc)
		},
	)
}
