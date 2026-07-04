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

package robytes

import (
	"sync"

	"github.com/samber/ro"
	"golang.org/x/text/cases"
	"golang.org/x/text/language"
)

// Dedicated English pools are cheaper than looking up the sync.Map each time.
var (
	englishTitleCaserPool = sync.Pool{New: func() any { c := cases.Title(language.English); return &c }}
	englishLowerCaserPool = sync.Pool{New: func() any { c := cases.Lower(language.English); return &c }}

	titleCaserPools sync.Map // map[string]*sync.Pool  (BCP 47 tag → pool of *cases.Caser)
	lowerCaserPools sync.Map // map[string]*sync.Pool
)

func acquireTitleCaser(tag language.Tag) (*sync.Pool, *cases.Caser) {
	key := tag.String()
	if v, ok := titleCaserPools.Load(key); ok {
		pool, _ := v.(*sync.Pool)
		c, _ := pool.Get().(*cases.Caser)
		return pool, c
	}
	p := &sync.Pool{New: func() any { c := cases.Title(tag); return &c }}
	actual, _ := titleCaserPools.LoadOrStore(key, p)
	pool, _ := actual.(*sync.Pool)
	c, _ := pool.Get().(*cases.Caser)
	return pool, c
}

func acquireLowerCaser(tag language.Tag) (*sync.Pool, *cases.Caser) {
	key := tag.String()
	if v, ok := lowerCaserPools.Load(key); ok {
		pool, _ := v.(*sync.Pool)
		c, _ := pool.Get().(*cases.Caser)
		return pool, c
	}
	p := &sync.Pool{New: func() any { c := cases.Lower(tag); return &c }}
	actual, _ := lowerCaserPools.LoadOrStore(key, p)
	pool, _ := actual.(*sync.Pool)
	c, _ := pool.Get().(*cases.Caser)
	return pool, c
}

func capitalize(str []byte) []byte {
	c, _ := englishTitleCaserPool.Get().(*cases.Caser)
	defer englishTitleCaserPool.Put(c)
	return c.Bytes(str)
}

func lowerEnglish(str []byte) []byte {
	c, _ := englishLowerCaserPool.Get().(*cases.Caser)
	defer englishLowerCaserPool.Put(c)
	return c.Bytes(str)
}

func capitalizeWithLanguage(str []byte, tag language.Tag) []byte {
	pool, c := acquireTitleCaser(tag)
	defer pool.Put(c)
	return c.Bytes(str)
}

// Capitalize capitalizes the first letter of the string.
// Play: https://go.dev/play/p/gAKIElJIUun
func Capitalize[T ~[]byte]() func(destination ro.Observable[T]) ro.Observable[T] {
	return ro.Map(
		func(value T) T {
			return T(capitalize(value))
		},
	)
}

// CapitalizeWithLanguage capitalizes the first letter of the byte slice using locale-aware casing.
func CapitalizeWithLanguage[T ~[]byte](tag language.Tag) func(destination ro.Observable[T]) ro.Observable[T] {
	return ro.Map(
		func(value T) T {
			return T(capitalizeWithLanguage(value, tag))
		},
	)
}
