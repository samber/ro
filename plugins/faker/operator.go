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

	"github.com/go-faker/faker/v4"
	"github.com/go-faker/faker/v4/pkg/options"
	"github.com/samber/ro"
)

// generate emits the values returned by generator, count times, or until the subscriber goes away
// when count is negative. The generator runs once per value and once per subscription, so every
// subscriber gets its own fresh values.
//
// It stops when the destination is closed, and fails with the context error when the subscriber
// context is cancelled. A generator error is forwarded to the destination and ends the stream.
func generate[T any](count int64, generator func() (T, error)) ro.Observable[T] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[T]) ro.Teardown {
		for i := int64(0); count < 0 || i < count; i++ {
			if destination.IsClosed() {
				return nil
			}

			if err := ctx.Err(); err != nil {
				destination.ErrorWithContext(ctx, err)
				return nil
			}

			value, err := generator()
			if err != nil {
				destination.ErrorWithContext(ctx, err)
				return nil
			}

			destination.NextWithContext(ctx, value)
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}

// generateString builds a finite string creation operator from one of the string helpers of go-faker.
func generateString(count int64, fn func(opts ...options.OptionFunc) string, opts []options.OptionFunc) ro.Observable[string] {
	if count < 0 {
		panic(ErrNegativeCount)
	}

	return generate(count, func() (string, error) {
		return fn(opts...), nil
	})
}

// Generate creates an Observable that emits an infinite sequence of the values returned by
// the generator, one call per value. Combine it with Take to bound it. A generator error is
// forwarded as an Error notification and ends the stream.
func Generate[T any](generator func() (T, error)) ro.Observable[T] {
	if generator == nil {
		panic(ErrNilGenerator)
	}

	return generate(-1, generator)
}

// GenerateN creates an Observable that emits count values returned by the generator, then completes.
// A generator error is forwarded as an Error notification and ends the stream.
func GenerateN[T any](count int64, generator func() (T, error)) ro.Observable[T] {
	if count < 0 {
		panic(ErrNegativeCount)
	}
	if generator == nil {
		panic(ErrNilGenerator)
	}

	return generate(count, generator)
}

// Fake creates an Observable that emits count values of type T filled with random data, then completes.
// Fields are populated by go-faker, driven by the `faker` struct tags of T. A tag error, such as
// an unknown tag or an invalid bound, is forwarded as an Error notification.
func Fake[T any](count int64, opts ...options.OptionFunc) ro.Observable[T] {
	if count < 0 {
		panic(ErrNegativeCount)
	}

	return generate(count, func() (T, error) {
		var value T
		err := faker.FakeData(&value, opts...)
		return value, err
	})
}

// Name emits count random full names.
func Name(count int64, opts ...options.OptionFunc) ro.Observable[string] {
	return generateString(count, faker.Name, opts)
}

// FirstName emits count random first names.
func FirstName(count int64, opts ...options.OptionFunc) ro.Observable[string] {
	return generateString(count, faker.FirstName, opts)
}

// LastName emits count random last names.
func LastName(count int64, opts ...options.OptionFunc) ro.Observable[string] {
	return generateString(count, faker.LastName, opts)
}

// Email emits count random email addresses.
func Email(count int64, opts ...options.OptionFunc) ro.Observable[string] {
	return generateString(count, faker.Email, opts)
}

// Username emits count random user names.
func Username(count int64, opts ...options.OptionFunc) ro.Observable[string] {
	return generateString(count, faker.Username, opts)
}

// PhoneNumber emits count random phone numbers.
func PhoneNumber(count int64, opts ...options.OptionFunc) ro.Observable[string] {
	return generateString(count, faker.Phonenumber, opts)
}

// UUID emits count random hyphenated UUIDs.
func UUID(count int64, opts ...options.OptionFunc) ro.Observable[string] {
	return generateString(count, faker.UUIDHyphenated, opts)
}

// IPv4 emits count random IPv4 addresses.
func IPv4(count int64, opts ...options.OptionFunc) ro.Observable[string] {
	return generateString(count, faker.IPv4, opts)
}

// IPv6 emits count random IPv6 addresses.
func IPv6(count int64, opts ...options.OptionFunc) ro.Observable[string] {
	return generateString(count, faker.IPv6, opts)
}

// URL emits count random URLs.
func URL(count int64, opts ...options.OptionFunc) ro.Observable[string] {
	return generateString(count, faker.URL, opts)
}

// Word emits count random lorem ipsum words.
func Word(count int64, opts ...options.OptionFunc) ro.Observable[string] {
	return generateString(count, faker.Word, opts)
}

// Sentence emits count random lorem ipsum sentences.
func Sentence(count int64, opts ...options.OptionFunc) ro.Observable[string] {
	return generateString(count, faker.Sentence, opts)
}

// Paragraph emits count random lorem ipsum paragraphs.
func Paragraph(count int64, opts ...options.OptionFunc) ro.Observable[string] {
	return generateString(count, faker.Paragraph, opts)
}
