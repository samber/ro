---
title: Introduction
description: Diagnose and resolve common issues with samber/ro
sidebar_position: 1
---

# 🔧 Troubleshooting Guide

This guide helps you diagnose and resolve common issues when working with reactive streams in `samber/ro`. Whether you're new to reactive programming or an experienced developer, you'll find practical solutions to the most frequently encountered problems.

## Quick Diagnostic Checklist

When something isn't working as expected, run through this quick checklist:

### 1. Verify Basic Stream Flow
```go
observable := ro.Pipe2(
    ro.Interval(1 * time.Second),
    ro.Take[int64](5),
    ro.Map(func(x int64) string {
        return fmt.Sprintf("Tick: %d", x)
    }),
)

// Add temporary debugging to see what's happening
observable = ro.Pipe3(
    observable,
    ro.TapOnNext(func(v T) { log.Printf("Next: %v", v) }),
    ro.TapOnError(func(e error) { log.Printf("Error: %v", e) }),
    ro.TapOnComplete(func() { log.Printf("Complete") }),
)
```

or

```go
observable := ro.Pipe2(
    ro.Interval(1 * time.Second),
    ro.Take[int64](5),
    ro.Map(func(x int64) string {
        return fmt.Sprintf("Tick: %d", x)
    }),
)

// Add temporary debugging to see what's happening
observable.Subscribe(ro.PrintObserver[string]())
```

### 2. Check Subscription Type
- Are you using `ro.OnNext()` or `ro.NoopObserver()` only? (ignores errors and completion)
- Are you expecting hot/cold behavior incorrectly?

### 3. Verify Context Usage
- Is your context being canceled unexpectedly?
- Are you passing context through the entire pipeline?

### 4. Examine Resource Management
- Are you properly cleaning up subscriptions?
- Are there goroutines or resources that aren't being released?

### 5. Test with Simple Data
Replace complex data sources with simple values:
```go
// Instead of: complexDatabaseQuery()
// Use: ro.Just(1, 2, 3) to test pipeline logic
```

Test each operators and then your pipeline with mocked data source:
```go
// foo.go
var pipeline = ro.PipeOp3(
    myOperator1,
    myOperator2,
    myOperator3,
)

func main() {
    observable := pipeline(mySource)
    observable.Subscribe(...)
}

// foo_test.go
func TestMyOperator1(t *testing.T) {
    values, err := ro.Collect(
        // deterministic data source
        myOperator1(ro.Just(1, 2, 3)),
    )
    // ...
}

func TestMyPipeline(t *testing.T) {
    values, err := ro.Collect(
        // deterministic data source
        pipeline(ro.Just(1, 2, 3)),
    )
    // ...
}
```

## When to Use This Guide

- **Stream not emitting values**: Check subscription and observable creation
- **Unexpected errors**: Review error handling and context propagation
- **Performance issues**: Look at memory usage and goroutine management
- **Memory leaks**: Verify resource cleanup in teardown functions
- **Concurrency problems**: Understand hot vs cold observables

## Common Symptom → Solution Mapping

| Symptom           | Likely Cause                             | First Steps                                       |
| ----------------- | ---------------------------------------- | ------------------------------------------------- |
| No values emitted | No subscription or completed observable  | Add logging, check subscriber                     |
| Immediate error   | Error in operator or source              | Add error handling, check source                  |
| High memory usage | Goroutine leak or unclosed subscription  | Profile with pprof, check cleanup                 |
| Slow performance  | Inefficient operators or backpressure    | Benchmark, optimize pipeline                      |
| Race conditions   | Unsafe observable with concurrent access | Use Safe observable or add the Serialize operator |

## Getting Help

If you can't resolve your issue using this guide:

1. **Create a Minimal Example**: Isolate the problem in a small, reproducible example
2. **Check GitHub Issues**: Search existing issues at https://github.com/samber/ro/issues
3. **File an Issue**: Create a new issue with:
   - Minimal reproduction code
   - Expected vs actual behavior
   - Go version and OS
   - Any relevant logs or error messages
   - A [Go playground](https://go.dev/play/) demo of the bug

## Debugging Philosophy

Reactive programming can be challenging because:
- **Asynchronous execution** makes traditional debugging harder
- **Stream composition** creates complex data flows
- **Context propagation** adds another layer of complexity

Follow these principles:
- **Start simple** - Test with basic data first
- **Add logging incrementally** - Don't overwhelm with debug output
- **Isolate components** - Test individual operators
- **Use the tools** - Leverage Go's debugging ecosystem

Ready to dive into specific issues? Choose a guide below:

- [**Common Issues**](./common-issues) - Frequently encountered problems
- [**Debugging Techniques**](./debugging) - Systematic debugging approaches  
- [**Performance Issues**](./performance) - Optimization and profiling
- [**Memory Leaks**](./memory-leaks) - Detection and prevention
- [**Concurrency Issues**](./concurrency) - Race conditions and synchronization
