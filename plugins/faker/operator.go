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
	"github.com/samber/ro"
)

// Fake creates an Observable that emits `count` fake values of type T.
// T should be a struct whose fields are annotated with `faker` struct tags
// for fine-grained control over generated values.
//
// Example:
//
//	type Person struct {
//	    Name  string `faker:"name"`
//	    Email string `faker:"email"`
//	}
//
//	obs := rofaker.Fake[Person](3)
func Fake[T any](count int) ro.Observable[T] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[T]) ro.Teardown {
		for i := 0; i < count; i++ {
			var v T
			if err := faker.FakeData(&v); err != nil {
				destination.ErrorWithContext(ctx, err)
				return nil
			}

			destination.NextWithContext(ctx, v)
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}

// Name creates an Observable that emits `count` random full names.
//
// Example:
//
//	obs := rofaker.Name(3)
func Name(count int) ro.Observable[string] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[string]) ro.Teardown {
		for i := 0; i < count; i++ {
			destination.NextWithContext(ctx, faker.Name())
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}

// FirstName creates an Observable that emits `count` random first names.
//
// Example:
//
//	obs := rofaker.FirstName(3)
func FirstName(count int) ro.Observable[string] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[string]) ro.Teardown {
		for i := 0; i < count; i++ {
			destination.NextWithContext(ctx, faker.FirstName())
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}

// LastName creates an Observable that emits `count` random last names.
//
// Example:
//
//	obs := rofaker.LastName(3)
func LastName(count int) ro.Observable[string] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[string]) ro.Teardown {
		for i := 0; i < count; i++ {
			destination.NextWithContext(ctx, faker.LastName())
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}

// Email creates an Observable that emits `count` random email addresses.
//
// Example:
//
//	obs := rofaker.Email(3)
func Email(count int) ro.Observable[string] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[string]) ro.Teardown {
		for i := 0; i < count; i++ {
			destination.NextWithContext(ctx, faker.Email())
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}

// PhoneNumber creates an Observable that emits `count` random phone numbers.
//
// Example:
//
//	obs := rofaker.PhoneNumber(3)
func PhoneNumber(count int) ro.Observable[string] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[string]) ro.Teardown {
		for i := 0; i < count; i++ {
			destination.NextWithContext(ctx, faker.Phonenumber())
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}

// URL creates an Observable that emits `count` random URLs.
//
// Example:
//
//	obs := rofaker.URL(3)
func URL(count int) ro.Observable[string] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[string]) ro.Teardown {
		for i := 0; i < count; i++ {
			destination.NextWithContext(ctx, faker.URL())
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}

// IPv4 creates an Observable that emits `count` random IPv4 addresses.
//
// Example:
//
//	obs := rofaker.IPv4(3)
func IPv4(count int) ro.Observable[string] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[string]) ro.Teardown {
		for i := 0; i < count; i++ {
			destination.NextWithContext(ctx, faker.IPv4())
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}

// IPv6 creates an Observable that emits `count` random IPv6 addresses.
//
// Example:
//
//	obs := rofaker.IPv6(3)
func IPv6(count int) ro.Observable[string] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[string]) ro.Teardown {
		for i := 0; i < count; i++ {
			destination.NextWithContext(ctx, faker.IPv6())
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}

// UUIDHyphenated creates an Observable that emits `count` random hyphenated UUIDs.
//
// Example:
//
//	obs := rofaker.UUIDHyphenated(3)
func UUIDHyphenated(count int) ro.Observable[string] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[string]) ro.Teardown {
		for i := 0; i < count; i++ {
			destination.NextWithContext(ctx, faker.UUIDHyphenated())
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}

// UUIDDigit creates an Observable that emits `count` random UUIDs without hyphens.
//
// Example:
//
//	obs := rofaker.UUIDDigit(3)
func UUIDDigit(count int) ro.Observable[string] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[string]) ro.Teardown {
		for i := 0; i < count; i++ {
			destination.NextWithContext(ctx, faker.UUIDDigit())
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}

// Word creates an Observable that emits `count` random Lorem Ipsum words.
//
// Example:
//
//	obs := rofaker.Word(3)
func Word(count int) ro.Observable[string] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[string]) ro.Teardown {
		for i := 0; i < count; i++ {
			destination.NextWithContext(ctx, faker.Word())
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}

// Sentence creates an Observable that emits `count` random Lorem Ipsum sentences.
//
// Example:
//
//	obs := rofaker.Sentence(3)
func Sentence(count int) ro.Observable[string] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[string]) ro.Teardown {
		for i := 0; i < count; i++ {
			destination.NextWithContext(ctx, faker.Sentence())
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}

// Paragraph creates an Observable that emits `count` random Lorem Ipsum paragraphs.
//
// Example:
//
//	obs := rofaker.Paragraph(3)
func Paragraph(count int) ro.Observable[string] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[string]) ro.Teardown {
		for i := 0; i < count; i++ {
			destination.NextWithContext(ctx, faker.Paragraph())
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}

// Latitude creates an Observable that emits `count` random latitude values.
//
// Example:
//
//	obs := rofaker.Latitude(3)
func Latitude(count int) ro.Observable[float64] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[float64]) ro.Teardown {
		for i := 0; i < count; i++ {
			destination.NextWithContext(ctx, faker.Latitude())
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}

// Longitude creates an Observable that emits `count` random longitude values.
//
// Example:
//
//	obs := rofaker.Longitude(3)
func Longitude(count int) ro.Observable[float64] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[float64]) ro.Teardown {
		for i := 0; i < count; i++ {
			destination.NextWithContext(ctx, faker.Longitude())
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}
