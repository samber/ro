package ro

import (
	"context"
	"testing"
)

func TestNewObserverUnsafe_panicsPropagate(t *testing.T) {
	t.Parallel()
	obs := NewUnsafeObserver[int](
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

func TestNewObserverWithContextUnsafe_panicsPropagate(t *testing.T) {
	t.Parallel()
	obs := NewObserverWithContextUnsafe[int](
		func(ctx context.Context, v int) { panic("boom") },
		func(ctx context.Context, err error) {},
		func(ctx context.Context) {},
	)

	recovered := false
	func() {
		defer func() {
			if r := recover(); r != nil {
				recovered = true
			}
		}()
		obs.NextWithContext(context.Background(), 1)
	}()

	if !recovered {
		t.Fatalf("expected panic to propagate from NewObserverWithContextUnsafe")
	}
}

func TestNewObserver_defaultCapturesPanic(t *testing.T) {
	t.Parallel()
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
