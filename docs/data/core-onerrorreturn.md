---
name: OnErrorReturn
slug: onerrorreturn
sourceRef: operator_error_handling.go#L108
type: core
category: error-handling
signatures:
  - "func OnErrorReturn[T any](finally T)"
playUrl: https://go.dev/play/p/d_9xe1oedjU
variantHelpers:
  - core#error-handling#onerrorreturn
similarHelpers:
  - core#error-handling#catch
  - core#error-handling#onerrorresumenextwith
position: 40
---

Emits a particular item when it encounters an error, then completes.

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.MapErr(func(i int) (int, error) {
        if i == 3 {
            return 0, errors.New("something went wrong")
        }
        return i, nil
    }),
    ro.OnErrorReturn(-1),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: -1 (error fallback)
// Completed
```

### With string fallback

```go
obs := ro.Pipe[string, string](
    ro.Just("apple", "banana", "invalid"),
    ro.MapErr(func(s string) (string, error) {
        if s == "invalid" {
            return "", errors.New("invalid fruit")
        }
        return strings.ToUpper(s), nil
    }),
    ro.OnErrorReturn("UNKNOWN"),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: "APPLE"
// Next: "BANANA"
// Next: "UNKNOWN" (error fallback)
// Completed
```

### API request with default value

```go
fetchUser := func(id int) Observable[string] {
    return ro.Defer(func() Observable[string] {
        if id == 999 {
            return ro.Throw[string](errors.New("user not found"))
        }
        return ro.Just(fmt.Sprintf("User%d", id))
    })
}

obs := ro.Pipe[string, string](
    fetchUser(999),
    ro.OnErrorReturn("Guest"),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: "Guest" (error fallback)
// Completed
```

### With multiple error handling

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.MapErr(func(i int) (int, error) {
        if i == 3 {
            return 0, fmt.Errorf("error at %d", i)
        }
        return i * 10, nil
    }),
    ro.OnErrorReturn(-999),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 10 (1*10)
// Next: 20 (2*10)
// Next: -999 (error fallback)
// Completed
```

### Configuration loading with default

```go
loadConfig := func() Observable[string] {
    return ro.Defer(func() Observable[string] {
        // Simulate config file not found
        return ro.Throw[string](errors.New("config.json not found"))
    })
}

obs := ro.Pipe[string, string](
    loadConfig(),
    ro.OnErrorReturn("default_config"),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: "default_config" (fallback to default)
// Completed
```

### With complex fallback object

```go
type User struct {
    ID   int
    Name string
}

fetchUser := func(id int) Observable[User] {
    return ro.Defer(func() Observable[User] {
        if id <= 0 {
            return ro.Throw[User](errors.New("invalid user ID"))
        }
        return User{ID: id, Name: fmt.Sprintf("User%d", id)}
    })
}

obs := ro.Pipe[User, User](
    fetchUser(-1),
    ro.OnErrorReturn(User{ID: 0, Name: "Anonymous"}),
)

sub := obs.Subscribe(ro.PrintObserver[User]())
defer sub.Unsubscribe()

// Next: {ID:0 Name:Anonymous} (error fallback)
// Completed
```

### In a processing pipeline

```go
processData := func(data []int) Observable[int] {
    return ro.Pipe[int, int](
        ro.FromSlice(data),
        ro.MapErr(func(i int) (int, error) {
            if i < 0 {
                return 0, fmt.Errorf("negative value: %d", i)
            }
            return i * 2, nil
        }),
        ro.OnErrorReturn(0), // Use 0 for negative values
    )
}

obs := processData([]int{1, 2, -3, 4})

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 2 (1*2)
// Next: 4 (2*2)
// Next: 0 (fallback for -3)
// Next: 8 (4*2)
// Completed
```