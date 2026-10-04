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

// Package fuzz contains race fuzz targets for the core package. Inputs encode the
// interleaving (seed, sizes, sync/async bitmask, early-stop choice), never expected results.
//
// Run with:
//
//	go test -race ./fuzz/
//
// See docs/docs/contributing.md#race-condition-patterns for the patterns they exercise.
package fuzz
