// Copyright 2025 samber.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://github.com/samber/ro/blob/main/licenses/LICENSE.apache.md
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package testing

import (
	stdtesting "testing"

	"github.com/samber/ro"
)

func BenchmarkMillionRowChallenge(b *stdtesting.B) {
	b.ReportAllocs()

	previous := ro.CaptureObserverPanics()
	ro.SetCaptureObserverPanics(false)
	b.Cleanup(func() {
		ro.SetCaptureObserverPanics(previous)
	})

	pipeline := ro.Pipe3(
		ro.Range(0, 1_000_000),
		ro.Map(func(value int64) int64 { return value + 1 }),
		ro.Filter(func(value int64) bool { return value%2 == 0 }),
		ro.Map(func(value int64) int64 { return value * 3 }),
	)

	const expectedSum int64 = 750001500000

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var sum int64

		subscription := pipeline.Subscribe(ro.NewObserver(
			func(value int64) {
				sum += value
			},
			func(err error) {
				b.Fatalf("unexpected error: %v", err)
			},
			func() {},
		))

		subscription.Wait()

		if sum != expectedSum {
			b.Fatalf("unexpected sum: %d", sum)
		}
	}
}
