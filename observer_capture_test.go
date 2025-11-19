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
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestObserverImpl_tryNextWithCapture_withCapture(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var errorCaught error
	observer := &observerImpl[int]{
		status:        0,
		capturePanics: true,
		onNext: func(ctx context.Context, value int) {
			panic("next panic")
		},
		onError: func(ctx context.Context, err error) {
			errorCaught = err
		},
		onComplete: func(ctx context.Context) {},
	}

	// Should capture the panic and call onError
	observer.tryNextWithCapture(context.Background(), 42, true)
	is.Error(errorCaught)
	is.Contains(errorCaught.Error(), "next panic")
}

func TestObserverImpl_tryNextWithCapture_withoutCapture(t *testing.T) {
	t.Parallel()

	observer := &observerImpl[int]{
		status:        0,
		capturePanics: false,
		onNext: func(ctx context.Context, value int) {
			panic("next panic")
		},
		onError:    func(ctx context.Context, err error) {},
		onComplete: func(ctx context.Context) {},
	}

	// Should propagate the panic
	recovered := false
	func() {
		defer func() {
			if r := recover(); r != nil {
				recovered = true
			}
		}()
		observer.tryNextWithCapture(context.Background(), 42, false)
	}()

	if !recovered {
		t.Fatalf("expected panic to propagate when capture=false")
	}
}

func TestObserverImpl_tryErrorWithCapture_withCapture(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var unhandledError error
	prev := GetOnUnhandledError()
	SetOnUnhandledError(func(ctx context.Context, err error) {
		unhandledError = err
	})
	defer SetOnUnhandledError(prev)

	observer := &observerImpl[int]{
		status:        0,
		capturePanics: true,
		onNext:        func(ctx context.Context, value int) {},
		onError: func(ctx context.Context, err error) {
			panic("error panic")
		},
		onComplete: func(ctx context.Context) {},
	}

	// Should capture the panic from onError and call OnUnhandledError
	observer.tryErrorWithCapture(context.Background(), assert.AnError, true)
	is.Error(unhandledError)
	is.Contains(unhandledError.Error(), "error panic")
}

func TestObserverImpl_tryErrorWithCapture_withoutCapture(t *testing.T) {
	t.Parallel()

	observer := &observerImpl[int]{
		status:        0,
		capturePanics: false,
		onNext:        func(ctx context.Context, value int) {},
		onError: func(ctx context.Context, err error) {
			panic("error panic")
		},
		onComplete: func(ctx context.Context) {},
	}

	// Should propagate the panic
	recovered := false
	func() {
		defer func() {
			if r := recover(); r != nil {
				recovered = true
			}
		}()
		observer.tryErrorWithCapture(context.Background(), assert.AnError, false)
	}()

	if !recovered {
		t.Fatalf("expected panic to propagate when capture=false")
	}
}

func TestObserverImpl_tryCompleteWithCapture_withCapture(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var unhandledError error
	prev := GetOnUnhandledError()
	SetOnUnhandledError(func(ctx context.Context, err error) {
		unhandledError = err
	})
	defer SetOnUnhandledError(prev)

	observer := &observerImpl[int]{
		status:        0,
		capturePanics: true,
		onNext:        func(ctx context.Context, value int) {},
		onError:       func(ctx context.Context, err error) {},
		onComplete: func(ctx context.Context) {
			panic("complete panic")
		},
	}

	// Should capture the panic from onComplete and call OnUnhandledError
	observer.tryCompleteWithCapture(context.Background(), true)
	is.Error(unhandledError)
	is.Contains(unhandledError.Error(), "complete panic")
}

func TestObserverImpl_tryCompleteWithCapture_withoutCapture(t *testing.T) {
	t.Parallel()

	observer := &observerImpl[int]{
		status:        0,
		capturePanics: false,
		onNext:        func(ctx context.Context, value int) {},
		onError:       func(ctx context.Context, err error) {},
		onComplete: func(ctx context.Context) {
			panic("complete panic")
		},
	}

	// Should propagate the panic
	recovered := false
	func() {
		defer func() {
			if r := recover(); r != nil {
				recovered = true
			}
		}()
		observer.tryCompleteWithCapture(context.Background(), false)
	}()

	if !recovered {
		t.Fatalf("expected panic to propagate when capture=false")
	}
}
