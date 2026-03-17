# ro/plugins/faker

A [samber/ro](https://github.com/samber/ro) plugin that provides creation operators for generating fake/random data using [go-faker/faker](https://github.com/go-faker/faker).

## Install

```bash
go get github.com/samber/ro/plugins/faker
```

## Usage

```go
import (
    "github.com/samber/ro"
    rofaker "github.com/samber/ro/plugins/faker"
)
```

## Operators

### `Fake[T any](count int) ro.Observable[T]`

Creates an Observable that emits `count` fake values of type T. The struct fields are populated using [faker struct tags](https://github.com/go-faker/faker#supported-tags).

```go
type Person struct {
    Name  string `faker:"name"`
    Email string `faker:"email"`
}

obs := rofaker.Fake[Person](5)
sub := obs.Subscribe(ro.PrintObserver[Person]())
defer sub.Unsubscribe()
```

### `Name(count int) ro.Observable[string]`

Creates an Observable that emits `count` random full names.

```go
obs := rofaker.Name(3)
sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()
// Next: John Smith
// Next: Jane Doe
// Next: Bob Johnson
// Completed
```

### `FirstName(count int) ro.Observable[string]`

Creates an Observable that emits `count` random first names.

### `LastName(count int) ro.Observable[string]`

Creates an Observable that emits `count` random last names.

### `Email(count int) ro.Observable[string]`

Creates an Observable that emits `count` random email addresses.

```go
obs := rofaker.Email(3)
sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()
// Next: john.smith@example.com
// Next: jane.doe@example.org
// Next: bob.johnson@example.net
// Completed
```

### `PhoneNumber(count int) ro.Observable[string]`

Creates an Observable that emits `count` random phone numbers.

### `URL(count int) ro.Observable[string]`

Creates an Observable that emits `count` random URLs.

### `IPv4(count int) ro.Observable[string]`

Creates an Observable that emits `count` random IPv4 addresses.

### `IPv6(count int) ro.Observable[string]`

Creates an Observable that emits `count` random IPv6 addresses.

### `UUIDHyphenated(count int) ro.Observable[string]`

Creates an Observable that emits `count` random hyphenated UUIDs (e.g., `550e8400-e29b-41d4-a716-446655440000`).

### `UUIDDigit(count int) ro.Observable[string]`

Creates an Observable that emits `count` random UUIDs without hyphens.

### `Word(count int) ro.Observable[string]`

Creates an Observable that emits `count` random Lorem Ipsum words.

### `Sentence(count int) ro.Observable[string]`

Creates an Observable that emits `count` random Lorem Ipsum sentences.

### `Paragraph(count int) ro.Observable[string]`

Creates an Observable that emits `count` random Lorem Ipsum paragraphs.

### `Latitude(count int) ro.Observable[float64]`

Creates an Observable that emits `count` random latitude values in the range [-90, 90].

### `Longitude(count int) ro.Observable[float64]`

Creates an Observable that emits `count` random longitude values in the range [-180, 180].

## Example: generating fake test data

```go
type User struct {
    ID    string `faker:"uuid_hyphenated"`
    Name  string `faker:"name"`
    Email string `faker:"email"`
    Phone string `faker:"phone_number"`
}

users, err := ro.Collect(rofaker.Fake[User](100))
if err != nil {
    log.Fatal(err)
}
```

## Composing with other operators

Creation operators can be composed with other ro operators using `ro.Pipe`:

```go
// Generate 10 fake emails and filter to those from .com domains
obs := ro.Pipe1(
    rofaker.Email(10),
    ro.Filter(func(email string) bool {
        return strings.HasSuffix(email, ".com")
    }),
)
```
