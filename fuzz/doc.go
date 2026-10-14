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

// Package fuzz contains race fuzz targets for the core package. Fuzz inputs encode the
// interleaving (sizes, sync or async source, when to stop), never the expected result.
//
// Run with:
//
//	go test -race ./fuzz/
//
// See docs/docs/contributing.md#race-condition-patterns for the patterns they exercise.
//
// # Writing a target
//
// A target reads top to bottom from its own file, without decoding anything:
//
//   - Name the file after the root source file it tests: operator_filter.go -> operator_filter_fuzz_test.go.
//   - Name the target FuzzOperator, or FuzzOperatorScenario when one operator has several scenarios.
//   - Write one doc comment above the target: the scenario, then the invariant asserted, then the seeds,
//     all in plain English.
//   - Give f.Fuzz explicit, typed arguments named by meaning: items uint8, asyncSource bool,
//     unsubscribeAfter uint8. Never pack bits into a mask, never decode one seed into several fields.
//   - Bound every argument inline with bounded, for example bounded(items, 0, maxItems).
//   - Keep one scenario per target. Prefer two targets (stop by Unsubscribe, stop by context
//     cancellation) to a bool or a mode that switches between them.
//   - Build the body from small named steps, in about 40 lines: build sources, build the pipeline, run it,
//     assert.
//   - Put helpers that only one operator needs in the same file, directly above their first use.
//     Put generic helpers in the purpose-named files listed below.
//   - Write identifiers in full words, without abbreviations or family prefixes. Comments explain why,
//     and stand alone.
//
// A complete target:
//
//	// FuzzFirst takes the first item >= threshold of a source of `items` integers...
//	func FuzzFirst(f *testing.F) {
//		fuzzSeeds(f, func(i int) []any {
//			// items, decisionIndex, asyncSource, sourceIgnoresStop, withIndex
//			return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%4 < 2, i%3 == 0}
//		})
//
//		f.Fuzz(func(t *testing.T, items, decisionIndex uint8, asyncSource, sourceIgnoresStop, withIndex bool) {
//			count := bounded(items, 0, maxItems)
//			decision := bounded(decisionIndex, 0, count+1)
//
//			numbers := newSource(count, asyncSource).ignoringStop(sourceIgnoresStop)
//			predicate := newCountingPredicate(true, func(item int) bool { return item >= decision })
//
//			got := collect(t, ro.First(predicate.test)(numbers.observable()), numbers)
//
//			got.expectValues(t, itemAt(decision, count))
//			got.expectCompletedOnce(t)
//			got.expectContract(t)
//			predicate.expectNoCallAfterDecision(t)
//		})
//	}
//
// # Sync and async sources
//
// A target that only uses synchronous sources is incomplete. A synchronous source emits inside Subscribe,
// so the teardown does not exist yet while items flow. An asynchronous source emits from its own
// goroutine, so Next races Unsubscribe and the terminal notification. Take the kind from a bool argument
// (asyncSource) and pass it to newSource. A target with several sources takes one bool per source.
//
// # Skipped targets
//
// A target that reproduces a known bug starts with
//
//	f.Skip("race: <id>; remove when fixed")
//
// The id names the bug and may be shared by several targets. Unskipping the target must reproduce
// the bug with the same failing assertion, and every target without a skip must pass. Do not weaken a
// target to make it pass.
//
// # Primitives
//
// seeds_test.go
//   - fuzzSeeds(f, generate): registers RO_FUZZ_ITERATIONS seeds, one per call of generate.
//   - seedByte(i, argumentIndex): the uint8 seed argument of seed i, spread over the byte range.
//   - bounded(value, low, high): maps any integer input into [low, high]. maxItems is the item limit.
//
// source_test.go
//   - newSource(items, async): emits 0..items-1 then completes, from Subscribe or from a goroutine.
//   - source.ignoringStop, failingAtEnd, neverEnding, startingAt, yieldingWith: refine a source.
//   - source.observable() gives the Observable; source.waitForProducers() waits for its goroutine.
//   - sequence(from, to), itemAt(index, count), smaller(a, b): expected values and arithmetic.
//
// recorder_test.go
//   - newRecorder[T](): destination that keeps values, errors, completions and late notifications.
//   - collect(t, observable, sources...): subscribes, waits for the sources, settles, returns the recorder.
//   - recorder.expectValues, expectAtMostValues, expectCompletedOnce, expectFailedOnce, expectContract.
//   - recorder.received, valueCount, terminalCount: live reads, safe from any goroutine.
//   - captureDroppedNotifications(t): counts notifications that a closed Subscriber absorbed.
//
// guards_test.go
//   - overlapGuard: enter and leave around a callback body, expectNone at the end.
//   - subscriptionCounter and countSubscriptions(counter, source): count live upstream subscriptions.
//     expectAllReleased waits for zero.
//
// predicate_test.go
//   - newCountingPredicate(decisiveResult, decide): predicate that counts calls after its deciding call.
//     Use .test, or .testIndexed for the I variants, and expectNoCallAfterDecision.
//
// stop_test.go
//   - subscribeInBackground(ctx, observable, recorder): Subscribe in a goroutine, so a hang fails.
//     Then expectReturn, unsubscribe, and unsubscribeAfterItems(t, recorder, items).
//   - cancelAfterItems(t, cancel, recorder, items): cancels the subscriber context after K items.
//   - To stop from inside the pipeline, use ro.Take.
//
// waits_test.go
//   - waitUntil(t, what, condition): bounded polling that fails with a clear message.
//   - runWithinDeadline(t, body): runs body in a goroutine, fails on panic or hang. body must not call t.
//   - waitDeadline and settleDelay: the bounds used by every wait.
//
// Assertions take the test as an argument and never store it, so a recorder can be shared with the
// goroutines of the pipeline.
package fuzz
