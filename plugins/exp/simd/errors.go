package simd

import "errors"

// Error definitions for the simd package.

var (
	// ErrClampLowerLessThanUpper is returned when Clamp functions are called
	// with a lower bound that is greater than the upper bound.
	ErrClampLowerLessThanUpper = errors.New("simd.Clamp: lower must be less than or equal to upper")
)
