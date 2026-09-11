---
name: Select
slug: select
sourceRef: plugins/exp/simd/int8.go#L213
type: plugin
category: simd
signatures:
  - "func (p PartialInt8s) Select(mask simd.Mask8s, other PartialInt8s) PartialInt8s"
  - "func (p PartialInt16s) Select(mask simd.Mask16s, other PartialInt16s) PartialInt16s"
  - "func (p PartialInt32s) Select(mask simd.Mask32s, other PartialInt32s) PartialInt32s"
  - "func (p PartialInt64s) Select(mask simd.Mask64s, other PartialInt64s) PartialInt64s"
  - "func (p PartialUint8s) Select(mask simd.Mask8s, other PartialUint8s) PartialUint8s"
  - "func (p PartialUint16s) Select(mask simd.Mask16s, other PartialUint16s) PartialUint16s"
  - "func (p PartialUint32s) Select(mask simd.Mask32s, other PartialUint32s) PartialUint32s"
  - "func (p PartialUint64s) Select(mask simd.Mask64s, other PartialUint64s) PartialUint64s"
  - "func (p PartialFloat32s) Select(mask simd.Mask32s, other PartialFloat32s) PartialFloat32s"
  - "func (p PartialFloat64s) Select(mask simd.Mask64s, other PartialFloat64s) PartialFloat64s"
playUrl:
variantHelpers:
  - plugin#simd#selectint8
  - plugin#simd#selectint16
  - plugin#simd#selectint32
  - plugin#simd#selectint64
  - plugin#simd#selectuint8
  - plugin#simd#selectuint16
  - plugin#simd#selectuint32
  - plugin#simd#selectuint64
  - plugin#simd#selectfloat32
  - plugin#simd#selectfloat64
similarHelpers:
  - plugin#simd#contains
  - plugin#simd#partial
position: 30
---

Takes each lane from the receiver where `mask` is set and from `other` where it is not, leaving padded lanes untouched.

It is the counterpart to `Contains`, whose mask can be handed straight back in — see that operator for a worked example.
