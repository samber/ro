package ro

import (
	"testing"
)

func TestNewObserverUnsafe_panicsPropagate(t *testing.T) {
	obs := NewObserverUnsafe[int](
		func(v int) { panic("boom") },
		func(err error) {},
		func() {},
	)

	recovered := false
	func() {
		defer func() {
			if r := recover(); r != nil {
				recovered = true
			}
		}()
		obs.Next(1)
	}()

	if !recovered {
		t.Fatalf("expected panic to propagate from NewObserverUnsafe")
	}
}

func TestNewObserver_defaultCapturesPanic(t *testing.T) {
	caught := false
	obs := NewObserver[int](
		func(v int) { panic("boom2") },
		func(err error) { caught = true },
		func() {},
	)

	// This should not panic; instead the onError handler should be called.
	obs.Next(1)

	if !caught {
		t.Fatalf("expected NewObserver to capture panic and call onError")
	}
}
