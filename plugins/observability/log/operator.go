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
	"log"

	"github.com/samber/ro"
)

func Log[T any]() func(ro.Observable[T]) ro.Observable[T] {
	return LogWithPrefix[T]("")
}

// Play: https://go.dev/play/p/plP2gfryjzK
func LogWithPrefix[T any](prefix string) func(ro.Observable[T]) ro.Observable[T] {
	if prefix != "" {
		prefix += " "
	}

	return ro.TapWithContext(
		func(ctx context.Context, value T) {
			// bearer:disable go_lang_logger_leak
			log.Printf("%sro.Next: %v", prefix, value)
		},
		func(ctx context.Context, err error) {
			// bearer:disable go_lang_logger_leak
			log.Printf("%sro.Error: %s", prefix, err.Error())
		},
		func(ctx context.Context) {
			// bearer:disable go_lang_logger_leak
			log.Printf("%sro.Complete", prefix)
		},
	)
}

// Play: https://go.dev/play/p/LuG085GGvvh
func FatalOnError[T any]() func(ro.Observable[T]) ro.Observable[T] {
	return FatalOnErrorWithPrefix[T]("")
}

// Play: https://go.dev/play/p/9E0dtrSJrxE
func FatalOnErrorWithPrefix[T any](prefix string) func(ro.Observable[T]) ro.Observable[T] {
	if prefix != "" {
		prefix += " "
	}

	return ro.TapOnErrorWithContext[T](
		func(ctx context.Context, err error) {
			// bearer:disable go_lang_logger_leak
			log.Fatalf("%sro.Error: %s", prefix, err.Error())
		},
	)
}
