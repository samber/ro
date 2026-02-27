---
title: SIMD (experimental)
description: High-performance slice operations using AVX, AVX2 and AVX512 SIMD when built with Go 1.26+ and GOEXPERIMENT=simd on amd64.
sidebar_position: 300
hide_table_of_contents: true
---

:::warning Help improve this documentation
This documentation is still new and evolving. If you spot any mistakes, unclear explanations, or missing details, please [open an issue](https://github.com/samber/ro/issues).

Your feedback helps us improve!
:::

#
## SIMD operators

This page lists all operators available in the `exp/simd` sub-package. These helpers use **AVX** (128-bit), **AVX2** (256-bit) or **AVX512** (512-bit) SIMD when built with Go 1.26+, the `GOEXPERIMENT=simd` flag, and on amd64.

:::warning Unstable API
SIMD operators are experimental. The API may break in the future.
:::

### Install

First, import the sub-package in your project:

```bash
go get -u github.com/samber/ro/plugins/exp/simd
```

import HelperList from '@site/plugins/helpers-pages/components/HelperList';

<HelperList 
  type="plugin"
  category="simd"
/>
