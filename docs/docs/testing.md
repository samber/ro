---
title: 🧪 Testing
description: Test your reactive pipelines
sidebar_position: 100
---

# Testing Reactive Pipelines

The `ro` library provides a comprehensive testing framework for unit testing reactive pipelines. This framework helps you verify that your [`Observable`](./core/observable) streams emit the expected sequence of values, errors, and completion signals.

The testing framework is designed to be intuitive and expressive, allowing you to write clear assertions about reactive streams without complex boilerplate code.

## Testing Package

The testing functionality is available in the `/testing` package:

```go
import rotesting "github.com/samber/ro/testing"
```

## Basic Usage

### Simple Assertion

The fluent API makes it easy to verify exact sequences of values. Each expectation builds on the previous one, creating readable and maintainable test code.

```go
import (
    "testing"
    "github.com/samber/ro"
    rotesting "github.com/samber/ro/testing"
)

func TestObservable(t *testing.T) {
    observable := ro.Just(1, 2, 3)

    rorotesting.Assert[int](t).
        Source(observable).
        ExpectNext(1).
        ExpectNext(2).
        ExpectNext(3).
        ExpectComplete().
        Verify()
}
```

### Mocking

Build reusable pipelines with `ro.PipeOpX(...)` variants, then invoke them with predictable test data. This pattern promotes clean separation between pipeline definition and testing.

```go
// Feature
var pipeline = ro.PipeOp3(
    ro.Filter(func(x int) int {
        return x%2 == 1
    })
    ro.Map(func(x int) string {
        return fmt.Sprintf("processed-%d", x)
    }),
    ro.DelayEach[string](100 * time.Millisecond)
)
```

```go
// Tests
func TestObservable(t *testing.T) {
    observable := pipeline(ro.Just(1, 2, 3, 4))

    rotesting.Assert[string](t).
        Source(observable).
        ExpectNext("processed-1").
        ExpectNext("processed-3").
        ExpectComplete().
        Verify()
}
```

### Testing Error Cases

Test error scenarios by creating observables that emit specific errors and verifying they're handled correctly. This is crucial for building robust reactive applications.

```go
func TestObservableError(t *testing.T) {
    observable := ro.Throw[string](errors.New("something went wrong"))

    rotesting.Assert[string](t).
        Source(observable).
        ExpectError(errors.New("something went wrong")).
        Verify()
}
```

### Sequence Assertions

Use `ExpectNextSeq()` to verify multiple values at once. This is more concise than multiple `ExpectNext()` calls and improves test readability.

```go
func TestObservableSequence(t *testing.T) {
    observable := ro.Range(1, 5)

    rotesting.Assert[int](t).
        Source(observable).
        ExpectNextSeq(1, 2, 3, 4, 5).
        ExpectComplete().
        Verify()
}
```

## API Reference

### AssertSpec Interface

The `AssertSpec[T]` interface provides a fluent API for testing observables:

```go
type AssertSpec[T any] interface {
    Source(source ro.Observable[T]) AssertSpec[T]
    ExpectNext(value T, msgAndArgs ...any) AssertSpec[T]
    ExpectNextSeq(items ...T) AssertSpec[T]
    ExpectError(err error, msgAndArgs ...any) AssertSpec[T]
    ExpectComplete(msgAndArgs ...any) AssertSpec[T]
    Verify()
    VerifyWithContext(ctx context.Context)
}
```

### Methods

#### `Source(source ro.Observable[T]) AssertSpec[T]`
Sets the observable to test.

#### `ExpectNext(value T, msgAndArgs ...any) AssertSpec[T]`
Expects the next value emitted by the observable. Optionally accepts custom error messages.

#### `ExpectNextSeq(items ...T) AssertSpec[T]`
Expects a sequence of values to be emitted by the observable.

#### `ExpectError(err error, msgAndArgs ...any) AssertSpec[T]`
Expects the observable to emit a specific error.

#### `ExpectComplete(msgAndArgs ...any) AssertSpec[T]`
Expects the observable to complete successfully.

#### `Verify()`
Subscribes to the observable and verifies all assertions.

#### `VerifyWithContext(ctx context.Context)`
Same as `Verify()` but with a custom context (eg: for timeout control). Context will be transmitted to the `.SubscribeWithContext(...)` method.

## Advanced Testing Patterns

### Testing with Context

Use context for timeout control and cancellation in time-based tests. This prevents infinite tests and mirrors real-world usage patterns.

```go
func TestObservableWithContext(t *testing.T) {
    ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
    defer cancel()

    observable := ro.Pipe1(
        ro.Interval(1 * time.Second),
        ro.Take(3),
    )

    rotesting.Assert[int](t).
        Source(observable).
        ExpectNextSeq(0, 1, 2).
        ExpectComplete().
        VerifyWithContext(ctx)
}
```

### Testing Custom Messages

Custom error messages make test failures easier to debug. When assertions fail, your custom messages will provide clear context about what went wrong.

```go
func TestObservableWithCustomMessages(t *testing.T) {
    observable := ro.Just("hello", "world")

    rotesting.Assert[string](t).
        Source(observable).
        ExpectNext("hello", "expected first value to be 'hello'").
        ExpectNext("world", "expected second value to be 'world'").
        ExpectComplete("expected observable to complete").
        Verify()
}
```

### Testing Hot Observables

Testing hot observables requires careful consideration of timing and concurrency. Use goroutines to simulate real-world scenarios where multiple subscribers receive values concurrently.

```go
func TestHotObservable(t *testing.T) {
    subject := ro.NewSubject[int]()

    // Start emitting values
    go func() {
        subject.Next(1)
        subject.Next(2)
        subject.Complete()
    }()

    rotesting.Assert[int](t).
        Source(subject).
        ExpectNextSeq(1, 2).
        ExpectComplete().
        Verify()
}
```

## Best Practices

Follow these practices to ensure comprehensive and maintainable test coverage for your reactive pipelines:

1. **Test Cold and Hot Observables**: Ensure your tests cover both cold (start on subscription) and hot (start immediately) observables.

2. **Use Context for Time-based Tests**: Always use context with timeout when testing time-based observables to prevent infinite tests.

3. **Test Error Cases**: Don't forget to test error scenarios in addition to success cases.

4. **Keep Tests Focused**: Each test should verify one specific behavior or scenario.

5. **Use Descriptive Messages**: Provide custom error messages to make test failures easier to understand.

## Integration with Testify

The `ro` library includes a testify plugin for enhanced assertion capabilities:

```go
import "github.com/samber/ro/plugins/testify"
```

This plugin provides additional helpers and integrations with the popular testify testing framework, making it easier to integrate reactive testing into existing Go codebases.

See [plugins documentation](./plugins/strings) for more details.
