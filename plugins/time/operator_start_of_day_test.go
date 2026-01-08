package rotime

import (
	"testing"
	"time"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

type timeTruncateTest struct {
	input    time.Time
	expected time.Time
}

var truncateDayTests = []timeTruncateTest{
	{
		input:    time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC),
		expected: time.Date(2026, time.January, 7, 0, 0, 0, 0, time.UTC),
	},
	{
		input:    time.Date(2025, time.December, 25, 23, 59, 59, 0, time.UTC),
		expected: time.Date(2025, time.December, 25, 0, 0, 0, 0, time.UTC),
	},
	{
		input:    time.Date(2026, time.January, 1, 0, 0, 0, 0, time.UTC),
		expected: time.Date(2026, time.January, 1, 0, 0, 0, 0, time.UTC),
	},
	{
		input:    time.Date(2026, time.January, 7, 14, 30, 0, 0, time.FixedZone("CET", 1*60*60)),
		expected: time.Date(2026, time.January, 7, 0, 0, 0, 0, time.FixedZone("CET", 1*60*60)),
	},
}

func TestStartOfDay(t *testing.T) {
	t.Run("Test Simple cases", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		for _, tt := range truncateDayTests {
			values, err := ro.Collect(
				ro.Pipe1(
					ro.Just(tt.input),
					StartOfDay(),
				),
			)
			is.Nil(err)
			is.Equal([]time.Time{tt.expected}, values)
		}
	})

	t.Run("Test empty obsersable case", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)
		values, err := ro.Collect(
			ro.Pipe1(
				ro.Empty[time.Time](),
				StartOfDay(),
			),
		)
		is.Equal([]time.Time{}, values)
		is.Nil(err)
	})

	t.Run("Test error handling case", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)
		values, err := ro.Collect(
			ro.Pipe1(
				ro.Throw[time.Time](assert.AnError),
				StartOfDay(),
			),
		)
		is.Equal([]time.Time{}, values)
		is.EqualError(err, assert.AnError.Error())
	})
}
