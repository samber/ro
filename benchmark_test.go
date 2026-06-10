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

package ro

import (
	"context"
	"fmt"
	"testing"
)

// The benchmarks below establish baselines for the per-event hot paths
// (Observer.Next, Subscriber.Next, operator chains, subjects) and for the
// buffering operators. They are fully synchronous so that goleak stays happy.
//
// Run with:
//
//	go test -run='^$' -bench='^Benchmark' -count=10 -benchmem github.com/samber/ro

func BenchmarkObserverNext(b *testing.B) {
	observer := NewObserver(
		func(value int) {},
		func(err error) {},
		func() {},
	)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		observer.Next(i)
	}
}

func BenchmarkObserverNextWithContext(b *testing.B) {
	ctx := context.Background()
	observer := NewObserverWithContext(
		func(ctx context.Context, value int) {},
		func(ctx context.Context, err error) {},
		func(ctx context.Context) {},
	)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		observer.NextWithContext(ctx, i)
	}
}

func BenchmarkSubscriberNext(b *testing.B) {
	modes := []struct {
		name string
		mode ConcurrencyMode
	}{
		{"safe", ConcurrencyModeSafe},
		{"unsafe", ConcurrencyModeUnsafe},
		{"eventually-safe", ConcurrencyModeEventuallySafe},
	}

	for _, m := range modes {
		b.Run(m.name, func(b *testing.B) {
			ctx := context.Background()
			subscriber := NewSubscriberWithConcurrencyMode(NoopObserver[int](), m.mode)

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				subscriber.NextWithContext(ctx, i)
			}
		})
	}
}

func BenchmarkPipeMapFilter(b *testing.B) {
	ctx := context.Background()
	subject := NewPublishSubject[int]()

	obs := Pipe2(
		subject.AsObservable(),
		Map(func(v int) int { return v * 2 }),
		Filter(func(v int) bool { return v%4 == 0 }),
	)

	sub := obs.Subscribe(NoopObserver[int]())
	defer sub.Unsubscribe()

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		subject.NextWithContext(ctx, i)
	}
}

func BenchmarkCollectRangePipe(b *testing.B) {
	obs := Pipe2(
		Range(0, 1000),
		Map(func(v int64) int64 { return v * 2 }),
		Filter(func(v int64) bool { return v%4 == 0 }),
	)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = Collect(obs)
	}
}

func BenchmarkTakeLast(b *testing.B) {
	for _, count := range []int{8, 128} {
		b.Run(fmt.Sprintf("count=%d", count), func(b *testing.B) {
			obs := TakeLast[int64](count)(Range(0, 1024))

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				_, _ = Collect(obs)
			}
		})
	}
}

func BenchmarkSkipLast(b *testing.B) {
	for _, count := range []int{8, 128} {
		b.Run(fmt.Sprintf("count=%d", count), func(b *testing.B) {
			obs := SkipLast[int64](count)(Range(0, 1024))

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				_, _ = Collect(obs)
			}
		})
	}
}

func BenchmarkZip2(b *testing.B) {
	obs := Zip2(Range(0, 256), Range(0, 256))

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = Collect(obs)
	}
}

func BenchmarkZipWith1(b *testing.B) {
	obs := ZipWith1[int64](Range(0, 256))(Range(0, 256))

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = Collect(obs)
	}
}

func BenchmarkPublishSubjectFanout(b *testing.B) {
	for _, observers := range []int{1, 8, 64} {
		b.Run(fmt.Sprintf("observers=%d", observers), func(b *testing.B) {
			ctx := context.Background()
			subject := NewPublishSubject[int]()

			subscriptions := make([]Subscription, 0, observers)
			for range make([]struct{}, observers) {
				subscriptions = append(subscriptions, subject.Subscribe(NoopObserver[int]()))
			}
			defer func() {
				for _, sub := range subscriptions {
					sub.Unsubscribe()
				}
			}()

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				subject.NextWithContext(ctx, i)
			}
		})
	}
}

func BenchmarkReplaySubjectBounded(b *testing.B) {
	ctx := context.Background()
	subject := NewReplaySubject[int](16)

	sub := subject.Subscribe(NoopObserver[int]())
	defer sub.Unsubscribe()

	b.ReportAllocs()
	b.ResetTimer()

	// Steady state: the buffer is full and every Next drops the oldest value.
	for i := 0; i < b.N; i++ {
		subject.NextWithContext(ctx, i)
	}
}

func BenchmarkReplaySubjectSubscribeReplay(b *testing.B) {
	ctx := context.Background()
	subject := NewReplaySubject[int](16)
	for i := 0; i < 32; i++ {
		subject.NextWithContext(ctx, i)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		subject.SubscribeWithContext(ctx, NoopObserver[int]()).Unsubscribe()
	}
}

func BenchmarkDistinct(b *testing.B) {
	ctx := context.Background()
	subject := NewPublishSubject[int]()

	obs := Distinct[int]()(subject.AsObservable())

	sub := obs.Subscribe(NoopObserver[int]())
	defer sub.Unsubscribe()

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		subject.NextWithContext(ctx, i%1024)
	}
}
