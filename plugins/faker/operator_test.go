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

package rofaker

import (
	"context"
	"net"
	"net/url"
	"regexp"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/go-faker/faker/v4/pkg/options"
	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

var uuidPattern = regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`)

type person struct {
	Name  string `faker:"name"`
	Email string `faker:"email"`
	ID    string `faker:"uuid_hyphenated"`
	Age   int    `faker:"boundary_start=18, boundary_end=65"`
}

type brokenTag struct {
	Name string `faker:"no_such_tag"`
}

func TestGenerate(t *testing.T) {
	t.Parallel()

	t.Run("infinite stream bounded by Take", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		var calls int64
		values, err := ro.Collect(
			ro.Pipe1(
				Generate(func() (int64, error) { return atomic.AddInt64(&calls, 1), nil }),
				ro.Take[int64](5),
			),
		)
		is.NoError(err)
		is.Equal([]int64{1, 2, 3, 4, 5}, values)
		// The generator is not called again once downstream is satisfied.
		is.Equal(int64(5), atomic.LoadInt64(&calls))
	})

	t.Run("generator error ends the stream", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		var calls int
		values, err := ro.Collect(
			Generate(func() (int, error) {
				calls++
				if calls == 3 {
					return 0, assert.AnError
				}
				return calls, nil
			}),
		)
		is.ErrorIs(err, assert.AnError)
		is.Equal([]int{1, 2}, values)
		is.Equal(3, calls)
	})

	t.Run("cancelled context fails the stream", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		ctx, cancel := context.WithCancel(context.Background())

		var calls int
		values, _, err := ro.CollectWithContext(
			ctx,
			Generate(func() (int, error) {
				calls++
				if calls == 4 {
					cancel()
				}
				return calls, nil
			}),
		)
		is.ErrorIs(err, context.Canceled)
		is.Equal([]int{1, 2, 3, 4}, values)
		is.Equal(4, calls)
	})

	t.Run("already cancelled context emits nothing", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		var calls int
		values, _, err := ro.CollectWithContext(
			ctx,
			GenerateN(10, func() (int, error) { calls++; return calls, nil }),
		)
		is.ErrorIs(err, context.Canceled)
		is.Empty(values)
		is.Zero(calls)
	})

	t.Run("each subscription gets its own values", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		var calls int64
		obs := GenerateN(3, func() (int64, error) { return atomic.AddInt64(&calls, 1), nil })

		first, err := ro.Collect(obs)
		is.NoError(err)
		second, err := ro.Collect(obs)
		is.NoError(err)

		is.Equal([]int64{1, 2, 3}, first)
		is.Equal([]int64{4, 5, 6}, second)
	})

	t.Run("nil generator panics", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		is.PanicsWithValue(ErrNilGenerator, func() { Generate[int](nil) })
		is.PanicsWithValue(ErrNilGenerator, func() { GenerateN[int](1, nil) })
	})
}

func TestGenerateN(t *testing.T) {
	t.Parallel()

	t.Run("emits count values then completes", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := ro.Collect(GenerateN(4, func() (string, error) { return "x", nil }))
		is.NoError(err)
		is.Equal([]string{"x", "x", "x", "x"}, values)
	})

	t.Run("zero count completes without calling the generator", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := ro.Collect(GenerateN(0, func() (string, error) {
			t.Error("generator must not be called")
			return "", nil
		}))
		is.NoError(err)
		is.Empty(values)
	})

	t.Run("negative count panics", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		is.PanicsWithValue(ErrNegativeCount, func() { GenerateN(-1, func() (int, error) { return 0, nil }) })
	})

	t.Run("early unsubscription stops the generator", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		var calls int64
		values, err := ro.Collect(ro.Pipe1(
			GenerateN(1_000_000, func() (int, error) { atomic.AddInt64(&calls, 1); return 0, nil }),
			ro.Take[int](2),
		))
		is.NoError(err)
		is.Len(values, 2)
		is.Equal(int64(2), atomic.LoadInt64(&calls))
	})
}

func TestFake(t *testing.T) {
	t.Parallel()

	t.Run("fills structs from their faker tags", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := ro.Collect(Fake[person](20))
		is.NoError(err)
		is.Len(values, 20)

		ids := map[string]struct{}{}
		for _, p := range values {
			is.NotEmpty(p.Name)
			is.Contains(p.Email, "@")
			is.Regexp(uuidPattern, p.ID)
			is.GreaterOrEqual(p.Age, 18)
			is.LessOrEqual(p.Age, 65)
			ids[p.ID] = struct{}{}
		}
		is.Len(ids, 20)
	})

	t.Run("forwards struct tag errors", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := ro.Collect(Fake[brokenTag](3))
		is.Error(err)
		is.Empty(values)
	})

	t.Run("honours options", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := ro.Collect(Fake[struct{ Label string }](5, options.WithRandomStringLength(7)))
		is.NoError(err)
		is.Len(values, 5)
		for _, v := range values {
			is.Len(v.Label, 7)
		}
	})

	t.Run("negative count panics", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		is.PanicsWithValue(ErrNegativeCount, func() { Fake[person](-1) })
	})
}

func TestStringOperators(t *testing.T) {
	t.Parallel()

	cases := map[string]struct {
		build func(count int64) ro.Observable[string]
		check func(is *assert.Assertions, value string)
	}{
		"Name":      {func(n int64) ro.Observable[string] { return Name(n) }, func(is *assert.Assertions, v string) { is.NotEmpty(v) }},
		"FirstName": {func(n int64) ro.Observable[string] { return FirstName(n) }, func(is *assert.Assertions, v string) { is.NotEmpty(v) }},
		"LastName":  {func(n int64) ro.Observable[string] { return LastName(n) }, func(is *assert.Assertions, v string) { is.NotEmpty(v) }},
		"Email":     {func(n int64) ro.Observable[string] { return Email(n) }, func(is *assert.Assertions, v string) { is.Contains(v, "@") }},
		"Username":  {func(n int64) ro.Observable[string] { return Username(n) }, func(is *assert.Assertions, v string) { is.NotEmpty(v) }},
		"PhoneNumber": {func(n int64) ro.Observable[string] { return PhoneNumber(n) }, func(is *assert.Assertions, v string) {
			is.NotEmpty(v)
		}},
		"UUID": {func(n int64) ro.Observable[string] { return UUID(n) }, func(is *assert.Assertions, v string) { is.Regexp(uuidPattern, v) }},
		"IPv4": {func(n int64) ro.Observable[string] { return IPv4(n) }, func(is *assert.Assertions, v string) {
			ip := net.ParseIP(v)
			is.NotNil(ip)
			is.NotNil(ip.To4())
		}},
		"IPv6": {func(n int64) ro.Observable[string] { return IPv6(n) }, func(is *assert.Assertions, v string) {
			ip := net.ParseIP(v)
			is.NotNil(ip)
			is.Contains(v, ":")
		}},
		"URL": {func(n int64) ro.Observable[string] { return URL(n) }, func(is *assert.Assertions, v string) {
			u, err := url.Parse(v)
			is.NoError(err)
			is.NotEmpty(u.Host)
		}},
		"Word":      {func(n int64) ro.Observable[string] { return Word(n) }, func(is *assert.Assertions, v string) { is.NotEmpty(v) }},
		"Sentence":  {func(n int64) ro.Observable[string] { return Sentence(n) }, func(is *assert.Assertions, v string) { is.NotEmpty(v) }},
		"Paragraph": {func(n int64) ro.Observable[string] { return Paragraph(n) }, func(is *assert.Assertions, v string) { is.NotEmpty(v) }},
	}

	for name, tc := range cases {
		name, tc := name, tc

		t.Run(name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			values, err := ro.Collect(tc.build(8))
			is.NoError(err)
			is.Len(values, 8)
			for _, v := range values {
				tc.check(is, v)
			}

			empty, err := ro.Collect(tc.build(0))
			is.NoError(err)
			is.Empty(empty)

			is.PanicsWithValue(ErrNegativeCount, func() { tc.build(-1) })
		})
	}

	t.Run("values differ between items", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := ro.Collect(UUID(50))
		is.NoError(err)
		seen := map[string]struct{}{}
		for _, v := range values {
			seen[v] = struct{}{}
		}
		is.Len(seen, 50)
	})

	t.Run("options are forwarded to go-faker", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := ro.Collect(Word(3, options.WithRandomStringLength(3)))
		is.NoError(err)
		is.Len(values, 3)
	})
}

func TestStringOperators_concurrentSubscriptions(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	obs := Email(50)

	var wg sync.WaitGroup
	results := make([][]string, 8)
	errs := make([]error, 8)
	for i := range results {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i], errs[i] = ro.Collect(obs)
		}(i)
	}
	wg.Wait()

	for i := range results {
		is.NoError(errs[i])
		is.Len(results[i], 50)
	}
	is.NotEqual(results[0], results[1])
}
