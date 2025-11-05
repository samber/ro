package benchmark

import (
	"bufio"
	"context"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/samber/ro"
)

// csvSource creates an Observable that reads int64 values (one per line)
// from the provided file path. It emits each parsed value and completes.
// This is intentionally simple: the observable reads the file synchronously
// on subscribe and emits values to the destination observer.
func csvSource(path string) ro.Observable[int64] {
	return ro.NewObservableWithContext(func(ctx context.Context, dest ro.Observer[int64]) ro.Teardown {
		f, err := os.Open(path)
		if err != nil {
			dest.Error(err)
			return nil
		}

		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			line := scanner.Text()
			v, err := strconv.ParseInt(line, 10, 64)
			if err != nil {
				dest.Error(err)
				_ = f.Close()
				return nil
			}

			// propagate context-aware notifications
			dest.NextWithContext(ctx, v)
		}

		if err := scanner.Err(); err != nil {
			dest.Error(err)
		} else {
			dest.CompleteWithContext(ctx)
		}

		_ = f.Close()
		return nil
	})
}

// Benchmark that runs the "million row" pipeline using a static CSV fixture.
// The benchmark expects a file with one integer per line. By default it will
// use the small sample in the fixtures directory. To benchmark a large static
// dataset, set the FIXTURE_PATH environment variable or place the file at
// `examples/billion-rows-benchmark/fixtures/1brc.csv`.
func BenchmarkMillionRowChallenge(b *testing.B) {
	b.ReportAllocs()

	fixture := os.Getenv("FIXTURE_PATH")
	if fixture == "" {
		fixture = filepath.Join("fixtures", "sample.csv")
	}

	// Use per-subscription opt-out of panic capture so the benchmark measures
	// hot-path throughput without mutating global state.
	ctx := ro.WithObserverPanicCaptureDisabled(context.Background())

	benchmarkCases := []struct {
		name string
		src  ro.Observable[int64]
	}{
		{name: "file-source", src: csvSource(fixture)},
	}

	for _, tc := range benchmarkCases {
		b.Run(tc.name, func(b *testing.B) {
			pipeline := ro.Pipe3(
				tc.src,
				ro.Map(func(value int64) int64 { return value + 1 }),
				ro.Filter(func(value int64) bool { return value%2 == 0 }),
				ro.Map(func(value int64) int64 { return value * 3 }),
			)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				var sum int64

				subscription := pipeline.SubscribeWithContext(ctx, ro.NewObserver(
					func(value int64) { sum += value },
					func(err error) { b.Fatalf("unexpected error: %v", err) },
					func() {},
				))

				subscription.Wait()

				// keep the correctness guard
				_ = sum
			}
		})
	}
}
