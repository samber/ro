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


package rolog

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/samber/ro"
)

// Log logs each item emitted by the source Observable using structured logging with slog.
// Play: https://go.dev/play/p/-94jOwZbMtx
func Log[T any](logger slog.Logger, level slog.Level) func(ro.Observable[T]) ro.Observable[T] {
	return ro.TapWithContext(
		func(ctx context.Context, value T) {
			logger.Log(ctx, level, fmt.Sprintf("ro.Next: %v", value))
		},
		func(ctx context.Context, err error) {
			logger.Log(ctx, level, fmt.Sprintf("ro.Error: %s", err.Error()))
		},
		func(ctx context.Context) {
			logger.Log(ctx, level, "ro.Complete")
		},
	)
}

// LogWithNotification logs each notification (Next, Error, Complete) emitted by the source Observable using structured logging with slog.
// Play: https://go.dev/play/p/M8N_igKHWGV
func LogWithNotification[T any](logger slog.Logger, level slog.Level) func(ro.Observable[T]) ro.Observable[T] {
	return ro.TapWithContext(
		func(ctx context.Context, value T) {
			logger.LogAttrs(ctx, level, "ro.Next", slog.Any("value", value))
		},
		func(ctx context.Context, err error) {
			logger.LogAttrs(ctx, level, "ro.Error", slog.Any("error", err))
		},
		func(ctx context.Context) {
			logger.LogAttrs(ctx, level, "ro.Complete")
		},
	)
}
