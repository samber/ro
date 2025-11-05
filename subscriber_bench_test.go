package ro

import (
	"testing"
)

// BenchmarkSubscriberNextPath compares the hot-path cost of calling Next for
// different concurrency modes:
// - Safe: real mutex
// - Unsafe: no-op mutex (method calls happen but do nothing)
// - SingleProducer: lockless fast-path (no Lock/Unlock calls)
//
// The benchmark disables observer panic-capture to reduce noise from the
// panic-recovery wrappers and focus measurements on synchronization costs.
func BenchmarkSubscriberNextPath(b *testing.B) {
	prev := CaptureObserverPanics()
	SetCaptureObserverPanics(false)
	defer SetCaptureObserverPanics(prev)

	cases := []struct {
		name string
		mode ConcurrencyMode
	}{
		{"Safe", ConcurrencyModeSafe},
		{"Unsafe", ConcurrencyModeUnsafe},
		{"SingleProducer", ConcurrencyModeSingleProducer},
	}

	for _, c := range cases {
		c := c
		b.Run(c.name, func(b *testing.B) {
			sub := NewSubscriberWithConcurrencyMode[int](NoopObserver[int](), c.mode)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sub.Next(i)
			}
		})
	}
}
