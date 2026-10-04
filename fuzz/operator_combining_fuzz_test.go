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
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/samber/lo"
	"github.com/samber/ro"
	"github.com/samber/ro/internal/xfuzz"
	"github.com/stretchr/testify/assert"
)

func FuzzMergeAll(f *testing.F) {
	addStreamSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		runHigherOrder(t, higherOrderKindMergeAll, seed, size, mask, k)
	})
}

func FuzzConcatAll(f *testing.F) {
	addStreamSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		runHigherOrder(t, higherOrderKindConcatAll, seed, size, mask, k)
	})
}

func FuzzConcatWith(f *testing.F) {
	addStreamSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		runHigherOrder(t, higherOrderKindConcatWith, seed, size, mask, k)
	})
}

func FuzzStartWith(f *testing.F) {
	f.Skip("race: startwith-unsafe-merge; remove when fixed") // Fails on main: overlapping notifications on the downstream observer: 1
	addStreamSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		h := newStreamHarness(t)
		mode, kk := decodeEarlyStop(k)

		var aC, bC activeCounter

		p := fuzzBound(size, 0, 3)
		na := fuzzBound(seed, 0, 4)
		nb := fuzzBound(seed/5, 0, 4)

		prefixes := make([]int, p)
		for i := range prefixes {
			prefixes[i] = -1 - i
		}

		a := trackSubscriptions(&aC, taggedSource(seed+1, 1, na, fuzzIsAsync(mask, 0), sourceEndComplete))
		b := trackSubscriptions(&bC, taggedSource(seed+2, 2, nb, fuzzIsAsync(mask, 1), sourceEndComplete))

		obs := ro.StartWith(prefixes...)(ro.Merge(a, b))

		h.start(applyEarlyStop(obs, mode, kk))
		h.settle(mode, kk)
		h.verify(&aC, &bC)

		got := h.rec.snapshot()
		for i := 0; i < minInt(p, len(got)); i++ {
			if got[i] != prefixes[i] {
				h.fail("prefix %d is %d, want %d", i, got[i], prefixes[i])
			}
		}

		total := p + na + nb

		switch mode {
		case earlyStopNone:
			if len(got) != total || h.rec.completeCount() != 1 {
				h.fail("got %d/%d values, completes=%d", len(got), total, h.rec.completeCount())
			}
		case earlyStopTake:
			if len(got) != minInt(kk, total) {
				h.fail("take(%d) delivered %d of %d", kk, len(got), total)
			}
		}
	})
}

// ---------------------------------------------------------------------------------------------
// Race / RaceWith
// ---------------------------------------------------------------------------------------------

func runRace(t *testing.T, with bool, seed, size int64, mask uint8) {
	t.Helper()

	h := newStreamHarness(t)
	m := fuzzBound(size, 2, 4)
	endless := mask&boundedLoopBit != 0

	end := sourceEndComplete
	if endless {
		end = sourceEndNever
	}

	counters := make([]activeCounter, m)
	srcs := make([]ro.Observable[int], m)
	ptrs := make([]*activeCounter, m)

	for i := range srcs {
		ptrs[i] = &counters[i]
		// At least one item, so that a loser of an endless race still has something to race with.
		srcs[i] = trackSubscriptions(ptrs[i], taggedSource(seed+int64(i), i, fuzzBound(seed+int64(i)*5, 1, 3), fuzzIsAsync(mask, i), end))
	}

	var obs ro.Observable[int]
	if with {
		obs = ro.RaceWith(srcs[1:]...)(srcs[0])
	} else {
		obs = ro.Race(srcs...)
	}

	h.start(obs)

	if endless {
		h.waitFor("the winner's first item", func() bool { return h.rec.nextCount() >= 1 || h.rec.terminals() > 0 })

		winner := h.rec.snapshot()[0] / sourceTagStride
		for i := range counters {
			i := i
			if i != winner {
				h.waitFor(fmt.Sprintf("loser %d (winner %d) to be unsubscribed", i, winner), func() bool { return ptrs[i].activeCount() == 0 })
			}
		}

		h.settle(earlyStopExternal, 1)
	} else {
		h.settle(earlyStopNone, 0)
	}

	h.verify(ptrs...)

	got := h.rec.snapshot()
	for i := range got {
		if got[i]/sourceTagStride != got[0]/sourceTagStride {
			h.fail("values of two sources mixed: %v", got)
		}
	}
}

func FuzzRaceWith(f *testing.F) {
	f.Skip("race: race-winner-not-retained; remove when fixed") // Fails on main: timeout waiting for: upstream 0 to drop to 0 active subscriptions
	addStreamSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, _ int64) {
		runRace(t, true, seed, size, mask)
	})
}

func FuzzRace(f *testing.F) {
	f.Skip("race: race-winner-not-retained; remove when fixed") // Fails on main: timeout waiting for: upstream 0 to drop to 0 active subscriptions
	addStreamSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, _ int64) {
		runRace(t, false, seed, size, mask)
	})
}

func FuzzPairwise(f *testing.F) {
	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := pickBoundaryValue(seed, 1, 0, fuzzMaxItems)
		up := &activeCounter{}

		runBoundaryOperator(t, "Pairwise", seed, mask&unsubscribeBit != 0,
			func() ro.Observable[[]int] {
				return ro.Pairwise[int]()(countedSource(up, seed, n, fuzzIsAsync(mask, 0)))
			},
			func(got [][]int) error {
				want := n - 1
				if want < 0 {
					want = 0
				}

				if len(got) != want {
					return fmt.Errorf("%d pairs, want %d", len(got), want)
				}

				for i, p := range got {
					if len(p) != 2 || p[0] != i || p[1] != i+1 {
						return fmt.Errorf("pair %d is %v, want [%d %d]", i, p, i, i+1)
					}
				}

				return nil
			}, up)
	})
}

func FuzzZip2(f *testing.F) {
	f.Skip("race: zip-pairs-delivered-out-of-order; remove when fixed")

	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		na, nb := pickBoundaryValue(seed, 1, 0, fuzzMaxItems), pickBoundaryValue(seed, 2, 0, fuzzMaxItems)
		a, b := &activeCounter{}, &activeCounter{}

		runBoundaryOperator(t, "Zip2", seed, mask&unsubscribeBit != 0,
			func() ro.Observable[lo.Tuple2[int, int]] {
				return ro.Zip2(countedSource(a, seed, na, fuzzIsAsync(mask, 0)), countedSource(b, seed+1, nb, fuzzIsAsync(mask, 1)))
			},
			func(got []lo.Tuple2[int, int]) error {
				want := na
				if nb < want {
					want = nb
				}

				if len(got) != want {
					return fmt.Errorf("%d pairs, want %d: %v", len(got), want, got)
				}

				for i, p := range got {
					if p.A != i || p.B != i {
						return fmt.Errorf("pair %d is %v, want (%d,%d): out of order in %v", i, p, i, i, got)
					}
				}

				return nil
			}, a, b)
	})
}

func FuzzZipVariadic(f *testing.F) {
	f.Skip("race: zip-pairs-delivered-out-of-order; remove when fixed")

	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		ns := []int{pickBoundaryValue(seed, 1, 0, fuzzMaxItems), pickBoundaryValue(seed, 2, 0, fuzzMaxItems), pickBoundaryValue(seed, 6, 0, fuzzMaxItems)}
		cs := []*activeCounter{{}, {}, {}}

		runBoundaryOperator(t, "Zip", seed, mask&unsubscribeBit != 0,
			func() ro.Observable[[]int] {
				srcs := make([]ro.Observable[int], len(ns))
				for i := range ns {
					srcs[i] = countedSource(cs[i], seed+int64(i), ns[i], fuzzIsAsync(mask, i))
				}

				return ro.Zip(srcs...)
			},
			func(got [][]int) error {
				want := ns[0]
				for _, n := range ns {
					if n < want {
						want = n
					}
				}

				if len(got) != want {
					return fmt.Errorf("%d tuples, want %d: %v", len(got), want, got)
				}

				for i, p := range got {
					if len(p) != len(ns) || p[0] != i || p[1] != i || p[2] != i {
						return fmt.Errorf("tuple %d is %v, want all %d: out of order in %v", i, p, i, got)
					}
				}

				return nil
			}, cs...)
	})
}

// FuzzCombineLatest2 checks that the last tuple holds the latest value of both sources.
func FuzzCombineLatest2(f *testing.F) {
	f.Skip("race: combinelatest-stale-last-tuple; remove when fixed")

	registerCombineLatest2Fuzz(f, "CombineLatest2/final", func(got []lo.Tuple2[int, int], na, nb int) error {
		if last := got[len(got)-1]; last.A != na-1 || last.B != nb-1 {
			return fmt.Errorf("stale last tuple %v, want (%d,%d)", last, na-1, nb-1)
		}

		return nil
	})
}

// FuzzCombineLatest2Order checks that tuples never go back to an older value of a source.
func FuzzCombineLatest2Order(f *testing.F) {
	f.Skip("race: combinelatest-out-of-order-tuples; remove when fixed")

	registerCombineLatest2Fuzz(f, "CombineLatest2/order", func(got []lo.Tuple2[int, int], _, _ int) error {
		for i := 1; i < len(got); i++ {
			if got[i].A < got[i-1].A || got[i].B < got[i-1].B {
				return fmt.Errorf("tuple went backwards: %v then %v", got[i-1], got[i])
			}
		}

		return nil
	})
}

// FuzzCombineLatest2Duplicate checks that no two consecutive tuples are identical.
func FuzzCombineLatest2Duplicate(f *testing.F) {
	f.Skip("race: combinelatest-duplicate-tuples; remove when fixed")

	registerCombineLatest2Fuzz(f, "CombineLatest2/duplicate", func(got []lo.Tuple2[int, int], _, _ int) error {
		for i := 1; i < len(got); i++ {
			if got[i] == got[i-1] {
				return fmt.Errorf("duplicate tuple %v", got[i])
			}
		}

		return nil
	})
}

func registerCombineLatest2Fuzz(f *testing.F, name string, extra func(got []lo.Tuple2[int, int], na, nb int) error) {
	f.Helper()
	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		na, nb := pickBoundaryValue(seed, 1, 0, fuzzMaxItems), pickBoundaryValue(seed, 2, 0, fuzzMaxItems)
		a, b := &activeCounter{}, &activeCounter{}

		runBoundaryOperator(t, name, seed, mask&unsubscribeBit != 0,
			func() ro.Observable[lo.Tuple2[int, int]] {
				return ro.CombineLatest2(countedSource(a, seed, na, fuzzIsAsync(mask, 0)), countedSource(b, seed+1, nb, fuzzIsAsync(mask, 1)))
			},
			func(got []lo.Tuple2[int, int]) error {
				if na == 0 || nb == 0 {
					if len(got) != 0 {
						return fmt.Errorf("emitted %v although a source is empty", got)
					}

					return nil
				}

				if len(got) == 0 {
					return errors.New("no tuple emitted although both sources emitted")
				}

				return extra(got, na, nb)
			}, a, b)
	})
}

func FuzzCombineLatestAll(f *testing.F) {
	f.Skip("race: combinelatestall-stale-last-tuple; remove when fixed")

	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		ns := []int{pickBoundaryValue(seed, 1, 1, fuzzMaxItems), pickBoundaryValue(seed, 2, 1, fuzzMaxItems), pickBoundaryValue(seed, 6, 1, fuzzMaxItems)}
		cs := []*activeCounter{{}, {}, {}}

		runBoundaryOperator(t, "CombineLatestAll", seed, mask&unsubscribeBit != 0,
			func() ro.Observable[[]int] {
				srcs := make([]ro.Observable[int], len(ns))
				for i := range ns {
					srcs[i] = countedSource(cs[i], seed+int64(i), ns[i], fuzzIsAsync(mask, i))
				}

				return ro.CombineLatestAll[int]()(ro.Just(srcs...))
			},
			func(got [][]int) error {
				if len(got) == 0 {
					return errors.New("no tuple emitted although every source emitted")
				}

				last := got[len(got)-1]
				for i, n := range ns {
					if len(last) != len(ns) || last[i] != n-1 {
						return fmt.Errorf("stale last tuple %v, want latest values of every source", last)
					}
				}

				return nil
			}, cs...)
	})
}

func FuzzOperatorCombiningZipFutureCompletion(f *testing.F) {
	// A source completing while another goroutine delivers the last pair must not drop it.
	// The window is a few instructions wide, so every seed is one more attempt to hit it.
	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i)} })

	f.Fuzz(func(t *testing.T, seed int64) {
		for _, variant := range fuzzZipVariants() {
			variant := variant
			t.Run(variant.name, func(t *testing.T) {
				testWithTimeout(t, 5*time.Second)
				is := assert.New(t)

				want := make([]int, variant.arity)
				for i := range want {
					want[i] = i
				}

				release := make(chan struct{})
				sources := make([]ro.Observable[int], variant.arity)
				for i := range sources {
					i := i
					sources[i] = ro.Future(func() (int, error) {
						<-release
						fuzzJitter(seed, i)
						return i, nil
					})
				}

				// Collect blocks until completion, so release the futures concurrently.
				go func() {
					fuzzJitter(seed, variant.arity)
					close(release)
				}()
				values, err := ro.Collect(variant.zip(sources))
				is.NoError(err)
				is.Equal([][]int{want}, values)
			})
		}
	})
}

// fuzzZipVariant wraps one zip operator behind a common shape, so a fuzz target runs against all of them.
type fuzzZipVariant struct {
	name  string
	arity int
	// zip emits each group of paired values as a slice, whatever the operator's output type.
	zip func(sources []ro.Observable[int]) ro.Observable[[]int]
}

func fuzzZipVariants() []fuzzZipVariant {
	return []fuzzZipVariant{
		{"Zip", 2, func(s []ro.Observable[int]) ro.Observable[[]int] { return ro.Zip(s...) }},
		{"ZipAll", 2, func(s []ro.Observable[int]) ro.Observable[[]int] { return ro.ZipAll[int]()(ro.Just(s...)) }},
		{"Zip2", 2, func(s []ro.Observable[int]) ro.Observable[[]int] {
			return ro.Map(func(v lo.Tuple2[int, int]) []int { return []int{v.A, v.B} })(ro.Zip2(s[0], s[1]))
		}},
		{"Zip3", 3, func(s []ro.Observable[int]) ro.Observable[[]int] {
			return ro.Map(func(v lo.Tuple3[int, int, int]) []int { return []int{v.A, v.B, v.C} })(ro.Zip3(s[0], s[1], s[2]))
		}},
		{"Zip4", 4, func(s []ro.Observable[int]) ro.Observable[[]int] {
			return ro.Map(func(v lo.Tuple4[int, int, int, int]) []int { return []int{v.A, v.B, v.C, v.D} })(ro.Zip4(s[0], s[1], s[2], s[3]))
		}},
		{"Zip5", 5, func(s []ro.Observable[int]) ro.Observable[[]int] {
			return ro.Map(func(v lo.Tuple5[int, int, int, int, int]) []int { return []int{v.A, v.B, v.C, v.D, v.E} })(ro.Zip5(s[0], s[1], s[2], s[3], s[4]))
		}},
		{"Zip6", 6, func(s []ro.Observable[int]) ro.Observable[[]int] {
			return ro.Map(func(v lo.Tuple6[int, int, int, int, int, int]) []int { return []int{v.A, v.B, v.C, v.D, v.E, v.F} })(ro.Zip6(s[0], s[1], s[2], s[3], s[4], s[5]))
		}},
	}
}
