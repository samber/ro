---
title: SIMD (experimental)
description: SIMD operators for ro — Go reactive streams. Run AVX/AVX2/AVX512 slice math on amd64 with Go 1.26+ and GOEXPERIMENT=simd, emitted as Observable values.
sidebar_position: 300
hide_table_of_contents: true
---

# SIMD - Plugin operators

This page lists all operators available in the `exp/simd` sub-package. These helpers use **AVX** (128-bit), **AVX2** (256-bit) or **AVX512** (512-bit) SIMD when built with Go 1.26+, the `GOEXPERIMENT=simd` flag, and on amd64.

:::warning Help improve this documentation
This documentation is still new and evolving. If you spot any mistakes, unclear explanations, or missing details, please [open an issue](https://github.com/samber/ro/issues).

Your feedback helps us improve!
:::

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
