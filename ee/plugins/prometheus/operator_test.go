// Copyright 2025 samber.
//
// Licensed as an Enterprise License (the "License"); you may not use
// this file except in compliance with the License. You may obtain
// a copy of the License at:
//
// https://github.com/samber/ro/blob/main/licenses/LICENSE.ee.md
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


package roprometheus

import (
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

func TestIncCounterOnNext(t *testing.T) {
	// t.Parallel()
	is := assert.New(t)

	bypassLicenseCheck = true

	var myCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name:        "my_counter",
			Help:        "My counter",
			ConstLabels: prometheus.Labels{"a": "b"},
		},
	)
	reg := prometheus.NewRegistry()
	reg.MustRegister(myCounter)

	obs := ro.Pipe2(
		ro.Just(1, 2, 3),
		ro.MapErr(func(i int) (int, error) {
			return i, nil
		}),
		IncCounterOnNext[int](myCounter),
	)
	_, _ = ro.Collect(obs)

	is.Equal(float64(3), testutil.ToFloat64(myCounter))
	expected := `
		# HELP my_counter My counter
		# TYPE my_counter counter
		my_counter{a="b"} 3
	`
	if err := testutil.CollectAndCompare(myCounter, strings.NewReader(expected), "my_counter"); err != nil {
		t.Errorf("unexpected collecting result:\n%s", err)
	}
}

func TestIncCounterOnError(t *testing.T) {
	// t.Parallel()
	is := assert.New(t)

	bypassLicenseCheck = true

	var myCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name:        "my_counter",
			Help:        "My counter",
			ConstLabels: prometheus.Labels{"a": "b"},
		},
	)
	reg := prometheus.NewRegistry()
	reg.MustRegister(myCounter)

	// 1- no error
	obs := ro.Pipe2(
		ro.Just(1, 2, 3),
		ro.MapErr(func(i int) (int, error) {
			return i, nil
		}),
		IncCounterOnError[int](myCounter),
	)
	_, _ = ro.Collect(obs)

	is.Equal(float64(0), testutil.ToFloat64(myCounter))
	expected := `
		# HELP my_counter My counter
		# TYPE my_counter counter
		my_counter{a="b"} 0
	`
	if err := testutil.CollectAndCompare(myCounter, strings.NewReader(expected), "my_counter"); err != nil {
		t.Errorf("unexpected collecting result:\n%s", err)
	}

	// 2- with error
	obs = ro.Pipe2(
		ro.Just(1, 2, 3),
		ro.MapErr(func(i int) (int, error) {
			return i, assert.AnError
		}),
		IncCounterOnError[int](myCounter),
	)
	_, _ = ro.Collect(obs)

	is.Equal(float64(1), testutil.ToFloat64(myCounter))
	expected = `
		# HELP my_counter My counter
		# TYPE my_counter counter
		my_counter{a="b"} 1
	`
	if err := testutil.CollectAndCompare(myCounter, strings.NewReader(expected), "my_counter"); err != nil {
		t.Errorf("unexpected collecting result:\n%s", err)
	}
}

func TestIncCounterOnComplete(t *testing.T) {
	// t.Parallel()
	is := assert.New(t)

	bypassLicenseCheck = true

	var myCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name:        "my_counter",
			Help:        "My counter",
			ConstLabels: prometheus.Labels{"a": "b"},
		},
	)
	reg := prometheus.NewRegistry()
	reg.MustRegister(myCounter)

	obs := ro.Pipe2(
		ro.Just(1, 2, 3),
		ro.MapErr(func(i int) (int, error) {
			return i, nil
		}),
		IncCounterOnComplete[int](myCounter),
	)
	_, _ = ro.Collect(obs)
	_, _ = ro.Collect(obs)

	is.Equal(float64(2), testutil.ToFloat64(myCounter))
	expected := `
		# HELP my_counter My counter
		# TYPE my_counter counter
		my_counter{a="b"} 2
	`
	if err := testutil.CollectAndCompare(myCounter, strings.NewReader(expected), "my_counter"); err != nil {
		t.Errorf("unexpected collecting result:\n%s", err)
	}
}

func TestIncCounterOnSubscription(t *testing.T) {
	// t.Parallel()
	is := assert.New(t)

	bypassLicenseCheck = true

	var myCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name:        "my_counter",
			Help:        "My counter",
			ConstLabels: prometheus.Labels{"a": "b"},
		},
	)
	reg := prometheus.NewRegistry()
	reg.MustRegister(myCounter)

	obs := ro.Pipe2(
		ro.Just(1, 2, 3),
		ro.MapErr(func(i int) (int, error) {
			return i, nil
		}),
		IncCounterOnSubscription[int](myCounter),
	)
	_, _ = ro.Collect(obs)
	_, _ = ro.Collect(obs)

	is.Equal(float64(2), testutil.ToFloat64(myCounter))
	expected := `
		# HELP my_counter My counter
		# TYPE my_counter counter
		my_counter{a="b"} 2
	`
	if err := testutil.CollectAndCompare(myCounter, strings.NewReader(expected), "my_counter"); err != nil {
		t.Errorf("unexpected collecting result:\n%s", err)
	}
}

func TestObserveNextLag(t *testing.T) {
	// t.Parallel()

	bypassLicenseCheck = true

	var myHistogram = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Name:        "my_histogram",
			Help:        "My histogram",
			ConstLabels: prometheus.Labels{"a": "b"},
			Buckets:     prometheus.DefBuckets,
		},
	)
	reg := prometheus.NewRegistry()
	reg.MustRegister(myHistogram)

	// Test with immediate processing (should have very low latency)
	obs := ro.Pipe2(
		ro.Just(0, 1, 2, 3),
		ObserveNextLag[int](myHistogram),
		ro.Map(func(i int) int {
			time.Sleep(time.Duration(i)*time.Second + 15*time.Millisecond)
			return i
		}),
	)
	_, _ = ro.Collect(obs)

	// Check that the histogram has observations
	expected := `
		# HELP my_histogram My histogram
		# TYPE my_histogram histogram
		my_histogram_bucket{a="b",le="0.005"} 0
		my_histogram_bucket{a="b",le="0.01"} 0
		my_histogram_bucket{a="b",le="0.025"} 1
		my_histogram_bucket{a="b",le="0.05"} 1
		my_histogram_bucket{a="b",le="0.1"} 1
		my_histogram_bucket{a="b",le="0.25"} 1
		my_histogram_bucket{a="b",le="0.5"} 1
		my_histogram_bucket{a="b",le="1"} 1
		my_histogram_bucket{a="b",le="2.5"} 3
		my_histogram_bucket{a="b",le="5"} 4
		my_histogram_bucket{a="b",le="10"} 4
		my_histogram_bucket{a="b",le="+Inf"} 4
		my_histogram_count{a="b"} 4
	`
	if err := testutil.CollectAndCompare(myHistogram, strings.NewReader(expected), "my_histogram_bucket"); err != nil {
		t.Errorf("unexpected collecting result:\n%s", err)
	}

	t.Run("observable that emits an error", func(t *testing.T) {
		// t.Parallel()

		bypassLicenseCheck = true

		var myHistogram = prometheus.NewHistogram(
			prometheus.HistogramOpts{
				Name:        "my_histogram",
				Help:        "My histogram",
				ConstLabels: prometheus.Labels{"a": "b"},
			},
		)
		reg := prometheus.NewRegistry()
		reg.MustRegister(myHistogram)

		// Test with observable that emits an error
		obs := ro.Pipe2(
			ro.Throw[int](assert.AnError),
			ObserveNextLag[int](myHistogram),
			ro.Map(func(i int) int {
				time.Sleep(time.Duration(i)*time.Second + 15*time.Millisecond)
				return i
			}),
		)
		_, err := ro.Collect(obs)
		assert.Error(t, err)
		assert.Equal(t, assert.AnError, err)

		// Should have observed 0 values since no Next() was called

		expected := `
			# HELP my_histogram My histogram
			# TYPE my_histogram histogram
			my_histogram_bucket{a="b",le="0.005"} 0
			my_histogram_bucket{a="b",le="0.01"} 0
			my_histogram_bucket{a="b",le="0.025"} 0
			my_histogram_bucket{a="b",le="0.05"} 0
			my_histogram_bucket{a="b",le="0.1"} 0
			my_histogram_bucket{a="b",le="0.25"} 0
			my_histogram_bucket{a="b",le="0.5"} 0
			my_histogram_bucket{a="b",le="1"} 0
			my_histogram_bucket{a="b",le="2.5"} 0
			my_histogram_bucket{a="b",le="5"} 0
			my_histogram_bucket{a="b",le="10"} 0
			my_histogram_bucket{a="b",le="+Inf"} 0
			my_histogram_count{a="b"} 0
		`
		if err := testutil.CollectAndCompare(myHistogram, strings.NewReader(expected), "my_histogram"); err != nil {
			t.Errorf("unexpected collecting result:\n%s", err)
		}
	})

	t.Run("empty observable", func(t *testing.T) {
		// t.Parallel()

		bypassLicenseCheck = true

		var myHistogram = prometheus.NewHistogram(
			prometheus.HistogramOpts{
				Name:        "my_histogram",
				Help:        "My histogram",
				ConstLabels: prometheus.Labels{"a": "b"},
			},
		)
		reg := prometheus.NewRegistry()
		reg.MustRegister(myHistogram)

		// Test with empty observable
		obs := ro.Pipe2(
			ro.Empty[int](),
			ObserveNextLag[int](myHistogram),
			ro.Map(func(i int) int {
				time.Sleep(time.Duration(i)*time.Second + 15*time.Millisecond)
				return i
			}),
		)
		_, _ = ro.Collect(obs)

		// Should have observed 0 values

		expected := `
			# HELP my_histogram My histogram
			# TYPE my_histogram histogram
			my_histogram_bucket{a="b",le="0.005"} 0
			my_histogram_bucket{a="b",le="0.01"} 0
			my_histogram_bucket{a="b",le="0.025"} 0
			my_histogram_bucket{a="b",le="0.05"} 0
			my_histogram_bucket{a="b",le="0.1"} 0
			my_histogram_bucket{a="b",le="0.25"} 0
			my_histogram_bucket{a="b",le="0.5"} 0
			my_histogram_bucket{a="b",le="1"} 0
			my_histogram_bucket{a="b",le="2.5"} 0
			my_histogram_bucket{a="b",le="5"} 0
			my_histogram_bucket{a="b",le="10"} 0
			my_histogram_bucket{a="b",le="+Inf"} 0
			my_histogram_count{a="b"} 0
		`
		if err := testutil.CollectAndCompare(myHistogram, strings.NewReader(expected), "my_histogram"); err != nil {
			t.Errorf("unexpected collecting result:\n%s", err)
		}
	})
}
