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
	"sync"
	"testing"
)

// droppedNotificationMu serializes test-time overrides of the package-level
// `OnDroppedNotification` hook so tests do not concurrently write the global
// variable and cause data races. Tests that need to temporarily replace the
// hook should use WithDroppedNotification.
var droppedNotificationMu sync.Mutex

// WithDroppedNotification temporarily sets `OnDroppedNotification` to the
// provided handler while executing fn. The previous handler is restored when
// fn returns. The helper serializes mutations using a mutex so concurrent
// test goroutines don't perform simultaneous writes to the global hook.
func WithDroppedNotification(t *testing.T, handler func(ctx context.Context, notification fmt.Stringer), fn func()) {
	t.Helper()

	droppedNotificationMu.Lock()
	prev := OnDroppedNotification
	OnDroppedNotification = handler

	// Ensure restore and unlock even if fn panics.
	defer func() {
		OnDroppedNotification = prev
		droppedNotificationMu.Unlock()
	}()

	fn()
}
