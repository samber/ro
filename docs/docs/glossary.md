---
id: glossary
title: 📚 Glossary
description: Reactive Programming glossary
sidebar_position: 300
---

# 📚 Glossary

This glossary provides comprehensive definitions for key concepts in reactive programming with `samber/ro`. Use this as a quick reference when exploring the documentation.

## Reactive Programming

:::tip Core Paradigm

A programming paradigm focused on event-driven applications, data streams and the propagation of change. This is the foundation that libraries like `samber/ro` are built upon.

:::

See [About](./about) for a detailed introduction to reactive programming concepts.

## Observable

A source that emits a sequence of data or events over time. This is the core building block of reactive programming in `samber/ro`.

Learn more about [Observables](./core/observable) in the basics guide.

## Observer

:::warning Consumer Pattern

An entity that subscribes to an [`Observable`](./core/observable) to receive updates or notifications. Observers handle the three types of events: Next, Error, and Complete.

:::

See [Observer documentation](./core/observer) for detailed implementation patterns.

## Subscription

:::danger Resource Management

Represents the relationship between an [`Observable`](./core/observable) and an [`Observer`](./core/observer), enabling the flow of data. Proper subscription management is crucial for preventing resource leaks.

:::

Learn about [Subscription management](./core/subscription) best practices.

## Subscriber
An entity that reacts to values, errors, or completion signals from an `Observable`.

## Subject

A hybrid type that acts as both an [`Observable`](./core/observable) and an [`Observer`](./core/observer), enabling multicasting. Subjects are essential for creating hot observables and shared streams.

Explore different [Subject types](./core/subject) and their use cases.

## Stream
A sequence of asynchronous events or data values emitted over time.

## Event Loop
A loop that waits for and dispatches events or messages in a reactive system.

## Hot vs Cold

:::tip Observable Classification

Classification of [`Observable`](./core/observable) based on whether emission occurs independently (hot) or per [`Subscription`](./core/subscription) (cold). Understanding this distinction is crucial for proper stream behavior.

:::

See [getting started guide](./getting-started#Hot-vs-Cold-Observables) for practical examples.

## Hot Observable

:::warning Shared Execution

An [`Observable`](./core/observable) that emits values regardless of subscriptions; shared among all subscribers. Useful for events like mouse clicks or sensor data.

:::

Learn how to create hot observables using [Subjects](./core/subject).

## Cold Observable

An [`Observable`](./core/observable) that starts emitting values only when a [`Subscriber`](./core/subscription) connects, producing a fresh sequence each time. This is the default behavior in `samber/ro`.

## Backpressure

:::danger Performance Critical

A strategy to handle situations where data is produced faster than it can be consumed. In `samber/ro`, backpressure is handled naturally through blocking behavior.

:::

See [Observer vs Go Channels](./core/observer#Observer-vs-Go-Channels) for backpressure implementation details.

## Operator

:::tip Stream Transformation

A function or method that transforms, filters, or combines data streams. Operators are the building blocks that make reactive programming powerful and expressive.

:::

Explore the complete [Operators reference](./operator/creation) and [usage guide](./core/operators).

## Multicasting
Sharing a single stream of data with multiple subscribers.

## Streams processing
The continuous processing of data as it flows through a system, often in real time and distributed fashion, allowing applications to react to events, transform data, and trigger actions immediately as data arrives. Unlike batch processing, which handles data in large chunks, stream processing works on individual events or small windows of data.

## Batch processing
A data processing approach where large volumes of data are collected and processed together as a single unit or “batch.” Unlike stream processing, which handles data continuously in real time, batch processing executes tasks on the accumulated data at scheduled intervals.

## Asynchronous
Execution that happens independently of the main program flow, often without blocking. `samber/ro` is mostly synchronous.

## Event-Driven

A programming style where changes in state or external events trigger the execution of code. This is central to reactive programming and `samber/ro`'s design.

Read more about reactive programming concepts in the [About](./about) section.

## Push Model

:::tip Data Flow Pattern

A data flow model where producers push updates to consumers automatically. This is where reactive programming library such as `samber/ro` sit.

:::

Compare with [Pull Model](#Pull-Model) and see [Observer vs Go Channels](./core/observer#Observer-vs-Go-Channels) for practical implementation.

## Pull Model
A model where consumers request data from producers when needed.

## Completion
A signal indicating that an Observable has finished emitting values.

## Error Handling

:::warning Application Stability

Mechanisms to manage errors that occur during the emission of data streams. Proper error handling is essential for building robust reactive applications.

:::

Learn about error handling patterns in the [troubleshooting guide](./troubleshooting).

## Replay
A technique where a stream retains past values and can replay them to new subscribers.

## Collector
Capturing the values of a stream so the main thread can immediately use it.

## Schedulers

Components controlling when and where stream events are emitted and observed. `samber/ro` has no scheduler, since Go offer first-class citizen concurrency.

This differs from other reactive libraries that require explicit scheduling for concurrency management.

## Concurrency
Executing multiple tasks simultaneously in a reactive system.

## Non-blocking
Designing operations so they don’t block the main execution thread.

## Transformation operators

:::tip Data Conversion

Converting one stream of data into another, often using operators. This is one of the most common operations in reactive programming pipelines.

:::

See [Operators guide](./core/operators) for examples and best practices.

## Filtering operators

Selecting specific data from a stream based on certain criteria. Essential for reducing data volume and focusing on relevant information.

Explore filtering operators in the [Operators reference](./operator/filtering).

## Composition operators

:::warning Complex Pipelines

Combining multiple streams or operations to create more complex reactive behavior. Composition is key to building sophisticated data processing pipelines.

:::

Learn about stream composition in the [getting started guide](./getting-started#Combining-Streams).
