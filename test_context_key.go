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

// ctxKey is an unexported type used for context keys in tests to avoid
// using a basic type (like string) directly as a context key which
// triggers linters such as revive's context-keys-type rule.
type ctxKey string

// testCtxKey is the key used by tests that need to attach a value to a
// context and later retrieve it. Keep it unexported and typed.
const testCtxKey ctxKey = "test"
