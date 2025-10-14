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


package rologrus

import (
	"context"

	"github.com/samber/ro"
	"github.com/sirupsen/logrus"
)

func Log[T any](logger *logrus.Logger, level logrus.Level) func(ro.Observable[T]) ro.Observable[T] {
	return ro.TapWithContext(
		func(ctx context.Context, value T) {
			logger.WithContext(ctx).Logf(level, "ro.Next: %v", value)
		},
		func(ctx context.Context, err error) {
			logger.WithContext(ctx).Logf(level, "ro.Error: %s", err.Error())
		},
		func(ctx context.Context) {
			logger.WithContext(ctx).Logf(level, "ro.Complete")
		},
	)
}

func LogWithNotification[T any](logger *logrus.Logger, level logrus.Level) func(ro.Observable[T]) ro.Observable[T] {
	return ro.TapWithContext(
		func(ctx context.Context, value T) {
			logger.WithContext(ctx).WithField("value", value).Logf(level, "ro.Next")
		},
		func(ctx context.Context, err error) {
			logger.WithContext(ctx).WithError(err).Fatal("ro.Error")
		},
		func(ctx context.Context) {
			logger.WithContext(ctx).Logf(level, "ro.Complete")
		},
	)
}

func FatalOnError[T any](logger *logrus.Logger) func(ro.Observable[T]) ro.Observable[T] {
	return ro.TapOnErrorWithContext[T](
		func(ctx context.Context, err error) {
			logger.WithContext(ctx).WithError(err).Fatal("ro.Error")
		},
	)
}
