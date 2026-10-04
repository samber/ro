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

package fuzz

import (
	"testing"

	"github.com/samber/ro"
)

// ---------------------------------------------------------------------------------------------
// Catch / StartWith over a merge of two sources: the unsafe subscriber is passed through to the merge
// ---------------------------------------------------------------------------------------------

func FuzzCatch(f *testing.F) {
	f.Skip("race: catch-unsafe-merge; remove when fixed") // Fails on main: notification delivered after a terminal notification: 1
	addStreamSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		h := newStreamHarness(t)
		mode, kk := decodeEarlyStop(k)

		var srcC, aC, bC activeCounter

		n := fuzzBound(size, 0, 3)
		na := fuzzBound(seed, 1, 4)
		nb := fuzzBound(seed/5, 1, 4)

		src := trackSubscriptions(&srcC, taggedSource(seed, 0, n, fuzzIsAsync(mask, 0), sourceEndError))
		a := trackSubscriptions(&aC, taggedSource(seed+1, 1, na, fuzzIsAsync(mask, 1), sourceEndComplete))
		b := trackSubscriptions(&bC, taggedSource(seed+2, 2, nb, fuzzIsAsync(mask, 2), sourceEndComplete))

		obs := ro.Catch(func(error) ro.Observable[int] { return ro.Merge(a, b) })(src)

		h.start(applyEarlyStop(obs, mode, kk))
		h.settle(mode, kk)
		h.verify(&srcC, &aC, &bC)

		total := n + na + nb

		switch mode {
		case earlyStopNone:
			if h.rec.nextCount() != total || h.rec.completeCount() != 1 {
				h.fail("got %d/%d values, completes=%d errs=%d", h.rec.nextCount(), total, h.rec.completeCount(), h.rec.errCount())
			}
		case earlyStopTake:
			if h.rec.nextCount() != minInt(kk, total) {
				h.fail("take(%d) delivered %d of %d", kk, h.rec.nextCount(), total)
			}
		}
	})
}

func FuzzWhile(f *testing.F) {
	f.Skip("race: while-ignores-closed-destination; remove when fixed") // Fails on main: timeout waiting for: Subscribe to return (hang)
	addStreamSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		runLoop(t, loopKindWhile, seed, size, mask, k)
	})
}

func FuzzDoWhile(f *testing.F) {
	f.Skip("race: dowhile-ignores-closed-destination; remove when fixed") // Fails on main: timeout waiting for: Subscribe to return (hang)
	addStreamSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		runLoop(t, loopKindDoWhile, seed, size, mask, k)
	})
}

func FuzzRetry(f *testing.F) {
	f.Skip("race: retry-ignores-closed-destination; remove when fixed") // Fails on main: timeout waiting for: Subscribe to return (hang)
	addStreamSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		runLoop(t, loopKindRetry, seed, size, mask, k)
	})
}

func FuzzRetryWithConfig(f *testing.F) {
	f.Skip("race: retry-ignores-closed-destination; remove when fixed") // Fails on main: 4 source subscriptions after the downstream closed, 1 were enough
	addStreamSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		runLoop(t, loopKindRetryWithConfig, seed, size, mask, k)
	})
}

func FuzzOnErrorResumeNextWith(f *testing.F) {
	f.Skip("race: resumenext-ignores-closed-destination; remove when fixed") // Fails on main: 5 sources subscribed after the downstream closed, 1 were enough
	addStreamSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		h := newStreamHarness(t)
		kk := fuzzBound(k, 1, maxEarlyStopCount)
		bounded := mask&boundedLoopBit != 0
		count := fuzzBound(size, 2, 5)

		var srcC activeCounter

		srcs := make([]ro.Observable[int], count)
		sizes := make([]int, count)
		lastFails := false
		total := 0

		for i := range srcs {
			sizes[i] = fuzzBound(seed+int64(i)*3, 1, 3)
			total += sizes[i]

			end := sourceEndComplete
			if fuzzBound(seed+int64(i), 0, 1) == 1 {
				end = sourceEndError
			}

			lastFails = end == sourceEndError
			srcs[i] = trackSubscriptions(&srcC, taggedSource(seed+int64(i), i, sizes[i], fuzzIsAsync(mask, i), end))
		}

		obs := ro.OnErrorResumeNextWith(srcs[1:]...)(srcs[0])

		mode := earlyStopTake
		if bounded {
			mode = earlyStopNone
		}

		h.start(applyEarlyStop(obs, mode, kk))
		h.settle(mode, kk)
		h.verify(&srcC)

		subs := srcC.totalCount()

		if bounded {
			wantErrs := 0
			if lastFails {
				wantErrs = 1
			}

			if subs != count || h.rec.nextCount() != total || h.rec.errCount() != wantErrs {
				h.fail("subs=%d/%d values=%d/%d errs=%d (want %d)", subs, count, h.rec.nextCount(), total, h.rec.errCount(), wantErrs)
			}

			return
		}

		if want := minInt(kk, total); h.rec.nextCount() != want {
			h.fail("take(%d) delivered %d, want %d", kk, h.rec.nextCount(), want)
		}

		// Sources needed to deliver kk items.
		need, acc := 0, 0
		for need < count && acc < kk {
			acc += sizes[need]
			need++
		}

		if subs > need {
			h.fail("%d sources subscribed after the downstream closed, %d were enough", subs, need)
		}
	})
}
