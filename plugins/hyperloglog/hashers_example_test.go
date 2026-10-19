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

package rohyperloglog

import (
	"fmt"

	"github.com/samber/ro"
)

func ExampleStringHash_FNV64a() {
	// Count distinct strings using FNV-1a 64-bit hash
	observable := ro.Pipe1(
		ro.Just("alice", "bob", "charlie", "alice", "bob", "david"),
		CountDistinct[string](8, false, StringHash.FNV64a()),
	)

	subscription := observable.Subscribe(ro.PrintObserver[uint64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 4
	// Completed
}

func ExampleStringHash_SHA256() {
	// Count distinct strings using SHA-256 hash
	observable := ro.Pipe1(
		ro.Just("alice", "bob", "charlie", "alice", "bob", "david"),
		CountDistinct[string](12, false, StringHash.SHA256()),
	)

	subscription := observable.Subscribe(ro.PrintObserver[uint64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 4
	// Completed
}

func ExampleStringHash_MD5() {
	// Count distinct strings using MD5 hash
	observable := ro.Pipe1(
		ro.Just("alice", "bob", "charlie", "alice", "bob", "david"),
		CountDistinct[string](10, false, StringHash.MD5()),
	)

	subscription := observable.Subscribe(ro.PrintObserver[uint64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 4
	// Completed
}

func ExampleStringHash_MapHash() {
	// Count distinct strings using maphash
	observable := ro.Pipe1(
		ro.Just("alice", "bob", "charlie", "alice", "bob", "david"),
		CountDistinct(8, false, StringHash.MapHash()), // non-deterministic
	)

	subscription := observable.Subscribe(ro.PrintObserver[uint64]())
	defer subscription.Unsubscribe()
}

func ExampleBytesHash_FNV64a() {
	// Count distinct byte slices using FNV-1a 64-bit hash
	// Note: []byte is not comparable, so we need to convert to string first
	observable := ro.Pipe1(
		ro.Just("alice", "bob", "charlie", "alice", "bob", "david"),
		CountDistinct[string](8, false, func(input string) uint64 {
			return BytesHash.FNV64a()([]byte(input))
		}),
	)

	subscription := observable.Subscribe(ro.PrintObserver[uint64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 4
	// Completed
}

func ExampleBytesHash_SHA256() {
	// Count distinct byte slices using SHA-256 hash
	// Note: []byte is not comparable, so we need to convert to string first
	observable := ro.Pipe1(
		ro.Just("alice", "bob", "charlie", "alice", "bob", "david"),
		CountDistinct[string](12, false, func(input string) uint64 {
			return BytesHash.SHA256()([]byte(input))
		}),
	)

	subscription := observable.Subscribe(ro.PrintObserver[uint64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 4
	// Completed
}

func ExampleCountDistinctReduce_withStringHash() {
	// Count distinct with incremental updates using StringHash
	observable := ro.Pipe1(
		ro.Just("alice", "bob", "charlie", "alice", "bob", "david"),
		CountDistinctReduce[string](8, false, StringHash.FNV64a()),
	)

	subscription := observable.Subscribe(ro.PrintObserver[uint64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 3
	// Next: 3
	// Next: 4
	// Completed
}

func Example_compareDifferentHashers() {
	// Compare different hash functions on the same data
	data := ro.Just("alice", "bob", "charlie", "alice", "bob", "david")

	// Using FNV-1a
	fnvObservable := ro.Pipe1(data, CountDistinct[string](8, false, StringHash.FNV64a()))
	fnvSub := fnvObservable.Subscribe(ro.NewObserver(
		func(value uint64) { fmt.Printf("FNV-1a: %d\n", value) },
		func(err error) { fmt.Printf("FNV-1a error: %v\n", err) },
		func() { fmt.Println("FNV-1a completed") },
	))
	defer fnvSub.Unsubscribe()

	// Using SHA-256
	shaObservable := ro.Pipe1(data, CountDistinct[string](8, false, StringHash.SHA256()))
	shaSub := shaObservable.Subscribe(ro.NewObserver(
		func(value uint64) { fmt.Printf("SHA-256: %d\n", value) },
		func(err error) { fmt.Printf("SHA-256 error: %v\n", err) },
		func() { fmt.Println("SHA-256 completed") },
	))
	defer shaSub.Unsubscribe()

	// Output:
	// FNV-1a: 4
	// FNV-1a completed
	// SHA-256: 4
	// SHA-256 completed
}

func Example_usingAllStringHashers() {
	// Demonstrate all available string hash functions
	data := ro.Just("alice", "bob", "charlie", "alice", "bob", "david")

	hashers := []struct {
		name string
		hash func(string) uint64
	}{
		{"FNV64a", StringHash.FNV64a()},
		{"FNV64", StringHash.FNV64()},
		{"FNV32a", StringHash.FNV32a()},
		{"FNV32", StringHash.FNV32()},
		{"SHA256", StringHash.SHA256()},
		{"SHA1", StringHash.SHA1()},
		{"SHA512", StringHash.SHA512()},
		{"MD5", StringHash.MD5()},
		// Note: MapHash is non-deterministic and results may vary
		{"MapHash", StringHash.MapHash()},
	}

	for _, h := range hashers {
		observable := ro.Pipe1(data, CountDistinct[string](8, false, h.hash))
		subscription := observable.Subscribe(ro.NewObserver(
			func(value uint64) { fmt.Printf("%s: %d\n", h.name, value) },
			func(err error) { fmt.Printf("%s error: %v\n", h.name, err) },
			func() { fmt.Printf("%s completed\n", h.name) },
		))
		defer subscription.Unsubscribe()
	}

	// Output:
	// FNV64a: 4
	// FNV64a completed
	// FNV64: 4
	// FNV64 completed
	// FNV32a: 1
	// FNV32a completed
	// FNV32: 1
	// FNV32 completed
	// SHA256: 4
	// SHA256 completed
	// SHA1: 4
	// SHA1 completed
	// SHA512: 4
	// SHA512 completed
	// MD5: 4
	// MD5 completed
	// MapHash: 6
	// MapHash completed
}
