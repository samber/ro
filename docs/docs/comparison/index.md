---
title: ro vs channels, iter, lo, RxGo, RxJS
description: How samber/ro compares to Go channels, the iter package, samber/lo, and other reactive libraries like RxGo and RxJS
sidebar_position: 0
hide_table_of_contents: true
---

# Comparisons

`ro` is one option among several ways to handle asynchronous and event-driven data in Go — and it isn't meant to replace everything else in your toolbox. These pages compare `ro` to the tools it's most often weighed against, so you can pick the right one for the problem in front of you:

- **Native Go concurrency** ([channels vs ro](./channels-vs-ro)): when a raw `chan` and a couple of goroutines are enough, and when composing operators pays off.
- **The standard library `iter` package** ([iter vs ro](./iter-vs-ro)): pull-based iteration vs push-based streams.
- **[samber/lo](./lo-vs-ro)**: `ro`'s sibling library — synchronous helpers for finite, in-memory collections. Most real applications use both together.
- **[RxGo](./rxgo-vs-ro)**: the historical ReactiveX implementation for Go.
- **[RxJS](./rxjs-vs-ro)**: for developers coming from the JavaScript/TypeScript reactive ecosystem.

## Reference

import DocCardList from '@theme/DocCardList';

<DocCardList />
