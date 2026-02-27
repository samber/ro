package rosimd

import "errors"

// Error definitions for the rosimd package.

var (
	// ErrClampLowerLessThanUpper is returned when Clamp functions are called
	// with a lower bound that is greater than the upper bound.
	ErrClampLowerLessThanUpper = errors.New("rosimd.Clamp: lower must be less than or equal to upper")
)
