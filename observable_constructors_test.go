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

func TestNewUnsafeObservable(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var values []int
	obs := NewUnsafeObservable(func(destination Observer[int]) Teardown {
		destination.Next(1)
		destination.Next(2)
		destination.Next(3)
		destination.Complete()
		return nil
	})

	sub := obs.Subscribe(NewObserver(
		func(value int) { values = append(values, value) },
		func(err error) { t.Fatalf("unexpected error: %v", err) },
		func() {},
	))

	sub.Wait()
	is.Equal([]int{1, 2, 3}, values)
}

func TestNewEventuallySafeObservable(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var values []int
	obs := NewEventuallySafeObservable(func(destination Observer[int]) Teardown {
		destination.Next(1)
		destination.Next(2)
		destination.Next(3)
		destination.Complete()
		return nil
	})

	sub := obs.Subscribe(NewObserver(
		func(value int) { values = append(values, value) },
		func(err error) { t.Fatalf("unexpected error: %v", err) },
		func() {},
	))

	sub.Wait()
	is.Equal([]int{1, 2, 3}, values)
}

func TestNewSingleProducerObservable(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var values []int
	obs := NewSingleProducerObservable(func(destination Observer[int]) Teardown {
		destination.Next(1)
		destination.Next(2)
		destination.Next(3)
		destination.Complete()
		return nil
	})

	sub := obs.Subscribe(NewObserver(
		func(value int) { values = append(values, value) },
		func(err error) { t.Fatalf("unexpected error: %v", err) },
		func() {},
	))

	sub.Wait()
	is.Equal([]int{1, 2, 3}, values)
}

func TestNewSingleProducerObservableWithContext(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var values []int
	var ctxReceived context.Context
	obs := NewSingleProducerObservableWithContext(func(ctx context.Context, destination Observer[int]) Teardown {
		ctxReceived = ctx
		destination.NextWithContext(ctx, 1)
		destination.NextWithContext(ctx, 2)
		destination.NextWithContext(ctx, 3)
		destination.CompleteWithContext(ctx)
		return nil
	})

	ctx := context.WithValue(context.Background(), "test", "value")
	sub := obs.SubscribeWithContext(ctx, NewObserver(
		func(value int) { values = append(values, value) },
		func(err error) { t.Fatalf("unexpected error: %v", err) },
		func() {},
	))

	sub.Wait()
	is.Equal([]int{1, 2, 3}, values)
	is.NotNil(ctxReceived)
	is.Equal("value", ctxReceived.Value("test"))
}

func TestNewEventuallySafeObservableWithContext(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var values []int
	var ctxReceived context.Context
	obs := NewEventuallySafeObservableWithContext(func(ctx context.Context, destination Observer[int]) Teardown {
		ctxReceived = ctx
		destination.NextWithContext(ctx, 1)
		destination.NextWithContext(ctx, 2)
		destination.NextWithContext(ctx, 3)
		destination.CompleteWithContext(ctx)
		return nil
	})

	ctx := context.WithValue(context.Background(), "test", "value")
	sub := obs.SubscribeWithContext(ctx, NewObserver(
		func(value int) { values = append(values, value) },
		func(err error) { t.Fatalf("unexpected error: %v", err) },
		func() {},
	))

	sub.Wait()
	is.Equal([]int{1, 2, 3}, values)
	is.NotNil(ctxReceived)
	is.Equal("value", ctxReceived.Value("test"))
}
