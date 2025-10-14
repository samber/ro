---
title: Operators
description: Declarative and composable API
hide_table_of_contents: true
---

The `ro` library provides all basic operators:
- **Creation operators**: The data source, usually the first argument of `ro.Pipe`
- **Chainable operators**: They filter, validate, transform, enrich... messages
  - **Transforming operators**: They transform items emitted by an `Observable`
  - **Filtering operators**: They selectively emit items from a source `Observable`
  - **Conditional operators**: Boolean operators
  - **Math and aggregation operators**: They perform basic math operations
  - **Error handling operators**: They help to recover from error notifications from an `Observable`
  - **Combining operators**: Combine multiple `Observable` into one
  - **Connectable operators**: Convert cold into hot `Observable`
  - **Other**: manipulation of context, utility, async scheduling...
- **Plugins**: External operators (mostly IOs and library wrappers)

## Reference

import DocCardList from '@theme/DocCardList';

<DocCardList />
