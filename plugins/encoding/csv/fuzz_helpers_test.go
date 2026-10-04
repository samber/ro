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

package rocsv

import (
	"testing"
	"time"

	"github.com/samber/ro/internal/xfuzz"
)

const (
	// fuzzWait bounds every wait so a deadlock fails the target instead of hanging the suite.
	fuzzWait = 5 * time.Second

	// fuzzMaxItems keeps one scenario fast while still spanning several hand-offs.
	fuzzMaxItems = 16

	// fuzzReadCap ends an "infinite" reader that was never told to stop, so a bug cannot spin forever.
	fuzzReadCap = 2_000

	// bigFieldSize exceeds csv.Writer's 4096-byte buffer, so every row forces a flush to the underlying writer.
	bigFieldSize = 5000

	// maskAsync selects a goroutine-fed source over a synchronous one.
	maskAsync = 1 << 0
)

func fuzzSeeds(f *testing.F) {
	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })
}

func waitDone(t *testing.T, ch <-chan struct{}, what string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(fuzzWait):
		t.Fatalf("%s: timed out after %s", what, fuzzWait)
	}
}
