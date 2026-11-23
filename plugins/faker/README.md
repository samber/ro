# Faker Plugin

The faker plugin provides creation operators that emit fake data, built on [go-faker/faker](https://github.com/go-faker/faker). It is handy for tests, demos and load generation.

## Installation

```bash
go get github.com/samber/ro/plugins/faker
```

## Operators

The named operators (`Name`, `FirstName`, `LastName`, `Email`, `Username`, `PhoneNumber`, `UUID`, `IPv4`, `IPv6`, `URL`, `Word`, `Sentence`, `Paragraph`) emit `count` strings and complete. A negative count panics with `ErrNegativeCount`. Trailing arguments are go-faker options.

```go
import (
    "github.com/samber/ro"
    rofaker "github.com/samber/ro/plugins/faker"
)

observable := rofaker.Email(3)

subscription := observable.Subscribe(ro.PrintObserver[string]())
defer subscription.Unsubscribe()

// Next: <random email>
// Next: <random email>
// Next: <random email>
// Completed
```

### Fake

Emits `count` structs filled by go-faker, driven by `faker` struct tags. A tag error is forwarded as an error notification.

```go
type User struct {
    Name  string `faker:"name"`
    Email string `faker:"email"`
    Age   int    `faker:"boundary_start=18, boundary_end=65"`
}

observable := rofaker.Fake[User](10)
```

### Generate and GenerateN

Wrap any generator function. `Generate` is infinite, so bound it with `ro.Take`. `GenerateN` emits `count` values. A generator error is forwarded and ends the stream.

```go
observable := ro.Pipe1(
    rofaker.Generate(func() (string, error) {
        return faker.UUIDHyphenated(), nil
    }),
    ro.Take[string](5),
)
```

## Behavior

- Every subscription calls the generator again, so subscribers never share values.
- Emission stops as soon as the subscriber unsubscribes, or when `Take` is satisfied.
- If the subscriber context is cancelled, the stream fails with the context error.
- go-faker uses a package level random source. Set a seeded one with `faker.SetRandomSource(faker.NewSafeSource(...))` for reproducible output.
