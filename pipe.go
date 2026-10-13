// Copyright 2025 samber.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://github.com/samber/ro/blob/main/licenses/LICENSE.apache.md
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package ro

import (
	"reflect"
)

// Pipe builds a composition of operators that will be chained to transform
// an observable stream. It provides a clean, declarative way to describe
// complex asynchronous operations.
//
// `PipeX()` should be favored over `Pipe()`, because it offers more type-safety.
//
// `PipeOp()` is the operator version of `Pipe()`.
func Pipe[First, Last any](source Observable[First], operators ...any) Observable[Last] {
	o := reflect.ValueOf(source)

	// Generic types vary per operator, so types are validated via reflection here,
	// once per Pipe() call — never on the message path. Prefer the PipeX variants
	// for full compile-time type safety.
	for _, operator := range operators {
		funcValue := reflect.ValueOf(operator)

		// check operator is a function with 1 input and 1 output
		if funcValue.Type().Kind() != reflect.Func || funcValue.Type().NumIn() != 1 || funcValue.Type().NumOut() != 1 {
			panic(newPipeError("%s is not an operator", funcValue.Type()))
		}
		// check operator input implements Observable[T]
		if funcValue.Type().In(0).Kind() != reflect.Interface {
			panic(newPipeError("%s does not implements Observable[T]", funcValue.Type().In(0)))
		}
		// check operator output implements Observable[T]
		if funcValue.Type().Out(0).Kind() != reflect.Interface {
			panic(newPipeError("%s does not implements Observable[T]", funcValue.Type().Out(0)))
		}
		// check operator input implements source Observable[T]
		if !o.Type().Implements(funcValue.Type().In(0)) {
			panic(newPipeError("%s does not implements %s", o.Type(), funcValue.Type().In(0)))
		}

		o = funcValue.Call([]reflect.Value{o})[0]
	}

	// check operator output implements destination Observable[T]
	mock := reflect.TypeOf((*Observable[Last])(nil)).Elem()
	if !o.Type().Implements(mock) {
		panic(newPipeError("%s does not implements %s", o.Type(), mock))
	}

	v, _ := o.Interface().(Observable[Last])

	return v
}

// Pipe1 is a typesafe 🎉 implementation of Pipe, that takes a source and 1 operator.
//
// `PipeOp1()` is the operator version of `Pipe1()`.
func Pipe1[A, B any](
	source Observable[A],
	operator1 func(Observable[A]) Observable[B],
) Observable[B] {
	return operator1(source)
}

// Pipe2 is a typesafe 🎉 implementation of Pipe, that takes a source and 2 operators.
//
// `PipeOp2()` is the operator version of `Pipe2()`.
func Pipe2[A, B, C any](
	source Observable[A],
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
) Observable[C] {
	return operator2(
		operator1(source),
	)
}

// Pipe3 is a typesafe 🎉 implementation of Pipe, that takes a source and 3 operators.
//
// `PipeOp3()` is the operator version of `Pipe3()`.
func Pipe3[A, B, C, D any](
	source Observable[A],
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
) Observable[D] {
	return operator3(
		operator2(
			operator1(source),
		),
	)
}

// Pipe4 is a typesafe 🎉 implementation of Pipe, that takes a source and 4 operators.
//
// `PipeOp4()` is the operator version of `Pipe4()`.
func Pipe4[A, B, C, D, E any](
	source Observable[A],
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
) Observable[E] {
	return operator4(
		operator3(
			operator2(
				operator1(source),
			),
		),
	)
}

// Pipe5 is a typesafe 🎉 implementation of Pipe, that takes a source and 5 operators.
//
// `PipeOp5()` is the operator version of `Pipe5()`.
func Pipe5[A, B, C, D, E, F any](
	source Observable[A],
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
) Observable[F] {
	return operator5(
		operator4(
			operator3(
				operator2(
					operator1(source),
				),
			),
		),
	)
}

// Pipe6 is a typesafe 🎉 implementation of Pipe, that takes a source and 6 operators.
//
// `PipeOp6()` is the operator version of `Pipe6()`.
func Pipe6[A, B, C, D, E, F, G any](
	source Observable[A],
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
) Observable[G] {
	return operator6(
		operator5(
			operator4(
				operator3(
					operator2(
						operator1(source),
					),
				),
			),
		),
	)
}

// Pipe7 is a typesafe 🎉 implementation of Pipe, that takes a source and 7 operators.
//
// `PipeOp7()` is the operator version of `Pipe7()`.
func Pipe7[A, B, C, D, E, F, G, H any](
	source Observable[A],
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
) Observable[H] {
	return operator7(
		operator6(
			operator5(
				operator4(
					operator3(
						operator2(
							operator1(source),
						),
					),
				),
			),
		),
	)
}

// Pipe8 is a typesafe 🎉 implementation of Pipe, that takes a source and 8 operators.
//
// `PipeOp8()` is the operator version of `Pipe8()`.
func Pipe8[A, B, C, D, E, F, G, H, I any](
	source Observable[A],
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
) Observable[I] {
	return operator8(
		operator7(
			operator6(
				operator5(
					operator4(
						operator3(
							operator2(
								operator1(source),
							),
						),
					),
				),
			),
		),
	)
}

// Pipe9 is a typesafe 🎉 implementation of Pipe, that takes a source and 9 operators.
//
// `PipeOp9()` is the operator version of `Pipe9()`.
func Pipe9[A, B, C, D, E, F, G, H, I, J any](
	source Observable[A],
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
) Observable[J] {
	return operator9(
		operator8(
			operator7(
				operator6(
					operator5(
						operator4(
							operator3(
								operator2(
									operator1(source),
								),
							),
						),
					),
				),
			),
		),
	)
}

// Pipe10 is a typesafe 🎉 implementation of Pipe, that takes a source and 10 operators.
//
// `PipeOp10()` is the operator version of `Pipe10()`.
func Pipe10[A, B, C, D, E, F, G, H, I, J, K any](
	source Observable[A],
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
) Observable[K] {
	return operator10(
		operator9(
			operator8(
				operator7(
					operator6(
						operator5(
							operator4(
								operator3(
									operator2(
										operator1(source),
									),
								),
							),
						),
					),
				),
			),
		),
	)
}

// Pipe11 is a typesafe 🎉 implementation of Pipe, that takes a source and 11 operators.
//
// `PipeOp11()` is the operator version of `Pipe11()`.
func Pipe11[A, B, C, D, E, F, G, H, I, J, K, L any](
	source Observable[A],
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
) Observable[L] {
	return operator11(
		operator10(
			operator9(
				operator8(
					operator7(
						operator6(
							operator5(
								operator4(
									operator3(
										operator2(
											operator1(source),
										),
									),
								),
							),
						),
					),
				),
			),
		),
	)
}

// Pipe12 is a typesafe 🎉 implementation of Pipe, that takes a source and 12 operators.
//
// `PipeOp12()` is the operator version of `Pipe12()`.
func Pipe12[A, B, C, D, E, F, G, H, I, J, K, L, M any](
	source Observable[A],
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
) Observable[M] {
	return operator12(
		operator11(
			operator10(
				operator9(
					operator8(
						operator7(
							operator6(
								operator5(
									operator4(
										operator3(
											operator2(
												operator1(source),
											),
										),
									),
								),
							),
						),
					),
				),
			),
		),
	)
}

// Pipe13 is a typesafe 🎉 implementation of Pipe, that takes a source and 13 operators.
//
// `PipeOp13()` is the operator version of `Pipe13()`.
func Pipe13[A, B, C, D, E, F, G, H, I, J, K, L, M, N any](
	source Observable[A],
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
) Observable[N] {
	return operator13(
		operator12(
			operator11(
				operator10(
					operator9(
						operator8(
							operator7(
								operator6(
									operator5(
										operator4(
											operator3(
												operator2(
													operator1(source),
												),
											),
										),
									),
								),
							),
						),
					),
				),
			),
		),
	)
}

// Pipe14 is a typesafe 🎉 implementation of Pipe, that takes a source and 14 operators.
//
// `PipeOp14()` is the operator version of `Pipe14()`.
func Pipe14[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O any](
	source Observable[A],
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
	operator14 func(Observable[N]) Observable[O],
) Observable[O] {
	return operator14(
		operator13(
			operator12(
				operator11(
					operator10(
						operator9(
							operator8(
								operator7(
									operator6(
										operator5(
											operator4(
												operator3(
													operator2(
														operator1(source),
													),
												),
											),
										),
									),
								),
							),
						),
					),
				),
			),
		),
	)
}

// Pipe15 is a typesafe 🎉 implementation of Pipe, that takes a source and 15 operators.
//
// `PipeOp15()` is the operator version of `Pipe15()`.
func Pipe15[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P any](
	source Observable[A],
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
	operator14 func(Observable[N]) Observable[O],
	operator15 func(Observable[O]) Observable[P],
) Observable[P] {
	return operator15(
		operator14(
			operator13(
				operator12(
					operator11(
						operator10(
							operator9(
								operator8(
									operator7(
										operator6(
											operator5(
												operator4(
													operator3(
														operator2(
															operator1(source),
														),
													),
												),
											),
										),
									),
								),
							),
						),
					),
				),
			),
		),
	)
}

// Pipe16 is a typesafe 🎉 implementation of Pipe, that takes a source and 16 operators.
//
// `PipeOp16()` is the operator version of `Pipe16()`.
func Pipe16[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q any](
	source Observable[A],
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
	operator14 func(Observable[N]) Observable[O],
	operator15 func(Observable[O]) Observable[P],
	operator16 func(Observable[P]) Observable[Q],
) Observable[Q] {
	return operator16(
		operator15(
			operator14(
				operator13(
					operator12(
						operator11(
							operator10(
								operator9(
									operator8(
										operator7(
											operator6(
												operator5(
													operator4(
														operator3(
															operator2(
																operator1(source),
															),
														),
													),
												),
											),
										),
									),
								),
							),
						),
					),
				),
			),
		),
	)
}

// Pipe17 is a typesafe 🎉 implementation of Pipe, that takes a source and 17 operators.
//
// `PipeOp17()` is the operator version of `Pipe17()`.
func Pipe17[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R any](
	source Observable[A],
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
	operator14 func(Observable[N]) Observable[O],
	operator15 func(Observable[O]) Observable[P],
	operator16 func(Observable[P]) Observable[Q],
	operator17 func(Observable[Q]) Observable[R],
) Observable[R] {
	return operator17(
		operator16(
			operator15(
				operator14(
					operator13(
						operator12(
							operator11(
								operator10(
									operator9(
										operator8(
											operator7(
												operator6(
													operator5(
														operator4(
															operator3(
																operator2(
																	operator1(source),
																),
															),
														),
													),
												),
											),
										),
									),
								),
							),
						),
					),
				),
			),
		),
	)
}

// Pipe18 is a typesafe 🎉 implementation of Pipe, that takes a source and 18 operators.
//
// `PipeOp18()` is the operator version of `Pipe18()`.
func Pipe18[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S any](
	source Observable[A],
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
	operator14 func(Observable[N]) Observable[O],
	operator15 func(Observable[O]) Observable[P],
	operator16 func(Observable[P]) Observable[Q],
	operator17 func(Observable[Q]) Observable[R],
	operator18 func(Observable[R]) Observable[S],
) Observable[S] {
	return operator18(
		operator17(
			operator16(
				operator15(
					operator14(
						operator13(
							operator12(
								operator11(
									operator10(
										operator9(
											operator8(
												operator7(
													operator6(
														operator5(
															operator4(
																operator3(
																	operator2(
																		operator1(source),
																	),
																),
															),
														),
													),
												),
											),
										),
									),
								),
							),
						),
					),
				),
			),
		),
	)
}

// Pipe19 is a typesafe 🎉 implementation of Pipe, that takes a source and 19 operators.
//
// `PipeOp19()` is the operator version of `Pipe19()`.
func Pipe19[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T any](
	source Observable[A],
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
	operator14 func(Observable[N]) Observable[O],
	operator15 func(Observable[O]) Observable[P],
	operator16 func(Observable[P]) Observable[Q],
	operator17 func(Observable[Q]) Observable[R],
	operator18 func(Observable[R]) Observable[S],
	operator19 func(Observable[S]) Observable[T],
) Observable[T] {
	return operator19(
		operator18(
			operator17(
				operator16(
					operator15(
						operator14(
							operator13(
								operator12(
									operator11(
										operator10(
											operator9(
												operator8(
													operator7(
														operator6(
															operator5(
																operator4(
																	operator3(
																		operator2(
																			operator1(source),
																		),
																	),
																),
															),
														),
													),
												),
											),
										),
									),
								),
							),
						),
					),
				),
			),
		),
	)
}

// Pipe20 is a typesafe 🎉 implementation of Pipe, that takes a source and 20 operators.
//
// `PipeOp20()` is the operator version of `Pipe20()`.
func Pipe20[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U any](
	source Observable[A],
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
	operator14 func(Observable[N]) Observable[O],
	operator15 func(Observable[O]) Observable[P],
	operator16 func(Observable[P]) Observable[Q],
	operator17 func(Observable[Q]) Observable[R],
	operator18 func(Observable[R]) Observable[S],
	operator19 func(Observable[S]) Observable[T],
	operator20 func(Observable[T]) Observable[U],
) Observable[U] {
	return operator20(
		operator19(
			operator18(
				operator17(
					operator16(
						operator15(
							operator14(
								operator13(
									operator12(
										operator11(
											operator10(
												operator9(
													operator8(
														operator7(
															operator6(
																operator5(
																	operator4(
																		operator3(
																			operator2(
																				operator1(source),
																			),
																		),
																	),
																),
															),
														),
													),
												),
											),
										),
									),
								),
							),
						),
					),
				),
			),
		),
	)
}

// Pipe21 is a typesafe 🎉 implementation of Pipe, that takes a source and 21 operators.
//
// `PipeOp21()` is the operator version of `Pipe21()`.
func Pipe21[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V any](
	source Observable[A],
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
	operator14 func(Observable[N]) Observable[O],
	operator15 func(Observable[O]) Observable[P],
	operator16 func(Observable[P]) Observable[Q],
	operator17 func(Observable[Q]) Observable[R],
	operator18 func(Observable[R]) Observable[S],
	operator19 func(Observable[S]) Observable[T],
	operator20 func(Observable[T]) Observable[U],
	operator21 func(Observable[U]) Observable[V],
) Observable[V] {
	return operator21(
		operator20(
			operator19(
				operator18(
					operator17(
						operator16(
							operator15(
								operator14(
									operator13(
										operator12(
											operator11(
												operator10(
													operator9(
														operator8(
															operator7(
																operator6(
																	operator5(
																		operator4(
																			operator3(
																				operator2(
																					operator1(source),
																				),
																			),
																		),
																	),
																),
															),
														),
													),
												),
											),
										),
									),
								),
							),
						),
					),
				),
			),
		),
	)
}

// Pipe22 is a typesafe 🎉 implementation of Pipe, that takes a source and 22 operators.
//
// `PipeOp22()` is the operator version of `Pipe22()`.
func Pipe22[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W any](
	source Observable[A],
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
	operator14 func(Observable[N]) Observable[O],
	operator15 func(Observable[O]) Observable[P],
	operator16 func(Observable[P]) Observable[Q],
	operator17 func(Observable[Q]) Observable[R],
	operator18 func(Observable[R]) Observable[S],
	operator19 func(Observable[S]) Observable[T],
	operator20 func(Observable[T]) Observable[U],
	operator21 func(Observable[U]) Observable[V],
	operator22 func(Observable[V]) Observable[W],
) Observable[W] {
	return operator22(
		operator21(
			operator20(
				operator19(
					operator18(
						operator17(
							operator16(
								operator15(
									operator14(
										operator13(
											operator12(
												operator11(
													operator10(
														operator9(
															operator8(
																operator7(
																	operator6(
																		operator5(
																			operator4(
																				operator3(
																					operator2(
																						operator1(source),
																					),
																				),
																			),
																		),
																	),
																),
															),
														),
													),
												),
											),
										),
									),
								),
							),
						),
					),
				),
			),
		),
	)
}

// Pipe23 is a typesafe 🎉 implementation of Pipe, that takes a source and 23 operators.
//
// `PipeOp23()` is the operator version of `Pipe23()`.
func Pipe23[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X any](
	source Observable[A],
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
	operator14 func(Observable[N]) Observable[O],
	operator15 func(Observable[O]) Observable[P],
	operator16 func(Observable[P]) Observable[Q],
	operator17 func(Observable[Q]) Observable[R],
	operator18 func(Observable[R]) Observable[S],
	operator19 func(Observable[S]) Observable[T],
	operator20 func(Observable[T]) Observable[U],
	operator21 func(Observable[U]) Observable[V],
	operator22 func(Observable[V]) Observable[W],
	operator23 func(Observable[W]) Observable[X],
) Observable[X] {
	return operator23(
		operator22(
			operator21(
				operator20(
					operator19(
						operator18(
							operator17(
								operator16(
									operator15(
										operator14(
											operator13(
												operator12(
													operator11(
														operator10(
															operator9(
																operator8(
																	operator7(
																		operator6(
																			operator5(
																				operator4(
																					operator3(
																						operator2(
																							operator1(source),
																						),
																					),
																				),
																			),
																		),
																	),
																),
															),
														),
													),
												),
											),
										),
									),
								),
							),
						),
					),
				),
			),
		),
	)
}

// Pipe24 is a typesafe 🎉 implementation of Pipe, that takes a source and 24 operators.
//
// `PipeOp24()` is the operator version of `Pipe24()`.
func Pipe24[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y any](
	source Observable[A],
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
	operator14 func(Observable[N]) Observable[O],
	operator15 func(Observable[O]) Observable[P],
	operator16 func(Observable[P]) Observable[Q],
	operator17 func(Observable[Q]) Observable[R],
	operator18 func(Observable[R]) Observable[S],
	operator19 func(Observable[S]) Observable[T],
	operator20 func(Observable[T]) Observable[U],
	operator21 func(Observable[U]) Observable[V],
	operator22 func(Observable[V]) Observable[W],
	operator23 func(Observable[W]) Observable[X],
	operator24 func(Observable[X]) Observable[Y],
) Observable[Y] {
	return operator24(
		operator23(
			operator22(
				operator21(
					operator20(
						operator19(
							operator18(
								operator17(
									operator16(
										operator15(
											operator14(
												operator13(
													operator12(
														operator11(
															operator10(
																operator9(
																	operator8(
																		operator7(
																			operator6(
																				operator5(
																					operator4(
																						operator3(
																							operator2(
																								operator1(source),
																							),
																						),
																					),
																				),
																			),
																		),
																	),
																),
															),
														),
													),
												),
											),
										),
									),
								),
							),
						),
					),
				),
			),
		),
	)
}

// Pipe25 is a typesafe 🎉 implementation of Pipe, that takes a source and 25 operators.
//
// `PipeOp25()` is the operator version of `Pipe25()`.
func Pipe25[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z any](
	source Observable[A],
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
	operator14 func(Observable[N]) Observable[O],
	operator15 func(Observable[O]) Observable[P],
	operator16 func(Observable[P]) Observable[Q],
	operator17 func(Observable[Q]) Observable[R],
	operator18 func(Observable[R]) Observable[S],
	operator19 func(Observable[S]) Observable[T],
	operator20 func(Observable[T]) Observable[U],
	operator21 func(Observable[U]) Observable[V],
	operator22 func(Observable[V]) Observable[W],
	operator23 func(Observable[W]) Observable[X],
	operator24 func(Observable[X]) Observable[Y],
	operator25 func(Observable[Y]) Observable[Z],
) Observable[Z] {
	return operator25(
		operator24(
			operator23(
				operator22(
					operator21(
						operator20(
							operator19(
								operator18(
									operator17(
										operator16(
											operator15(
												operator14(
													operator13(
														operator12(
															operator11(
																operator10(
																	operator9(
																		operator8(
																			operator7(
																				operator6(
																					operator5(
																						operator4(
																							operator3(
																								operator2(
																									operator1(source),
																								),
																							),
																						),
																					),
																				),
																			),
																		),
																	),
																),
															),
														),
													),
												),
											),
										),
									),
								),
							),
						),
					),
				),
			),
		),
	)
}

// PipeOp is similar to Pipe, but can be used as an operator.
func PipeOp[First, Last any](operators ...any) func(Observable[First]) Observable[Last] {
	return func(source Observable[First]) Observable[Last] {
		return Pipe[First, Last](source, operators...)
	}
}

// PipeOp1 is similar to Pipe1, but can be used as an operator.
func PipeOp1[A, B any](
	operator1 func(Observable[A]) Observable[B],
) func(Observable[A]) Observable[B] {
	return func(source Observable[A]) Observable[B] {
		return Pipe1(
			source,
			operator1,
		)
	}
}

// PipeOp2 is similar to Pipe2, but can be used as an operator.
func PipeOp2[A, B, C any](
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
) func(Observable[A]) Observable[C] {
	return func(source Observable[A]) Observable[C] {
		return Pipe2(
			source,
			operator1,
			operator2,
		)
	}
}

// PipeOp3 is similar to Pipe3, but can be used as an operator.
func PipeOp3[A, B, C, D any](
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
) func(Observable[A]) Observable[D] {
	return func(source Observable[A]) Observable[D] {
		return Pipe3(
			source,
			operator1,
			operator2,
			operator3,
		)
	}
}

// PipeOp4 is similar to Pipe4, but can be used as an operator.
func PipeOp4[A, B, C, D, E any](
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
) func(Observable[A]) Observable[E] {
	return func(source Observable[A]) Observable[E] {
		return Pipe4(
			source,
			operator1,
			operator2,
			operator3,
			operator4,
		)
	}
}

// PipeOp5 is similar to Pipe5, but can be used as an operator.
func PipeOp5[A, B, C, D, E, F any](
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
) func(Observable[A]) Observable[F] {
	return func(source Observable[A]) Observable[F] {
		return Pipe5(
			source,
			operator1,
			operator2,
			operator3,
			operator4,
			operator5,
		)
	}
}

// PipeOp6 is similar to Pipe6, but can be used as an operator.
func PipeOp6[A, B, C, D, E, F, G any](
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
) func(Observable[A]) Observable[G] {
	return func(source Observable[A]) Observable[G] {
		return Pipe6(
			source,
			operator1,
			operator2,
			operator3,
			operator4,
			operator5,
			operator6,
		)
	}
}

// PipeOp7 is similar to Pipe7, but can be used as an operator.
func PipeOp7[A, B, C, D, E, F, G, H any](
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
) func(Observable[A]) Observable[H] {
	return func(source Observable[A]) Observable[H] {
		return Pipe7(
			source,
			operator1,
			operator2,
			operator3,
			operator4,
			operator5,
			operator6,
			operator7,
		)
	}
}

// PipeOp8 is similar to Pipe8, but can be used as an operator.
func PipeOp8[A, B, C, D, E, F, G, H, I any](
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
) func(Observable[A]) Observable[I] {
	return func(source Observable[A]) Observable[I] {
		return Pipe8(
			source,
			operator1,
			operator2,
			operator3,
			operator4,
			operator5,
			operator6,
			operator7,
			operator8,
		)
	}
}

// PipeOp9 is similar to Pipe9, but can be used as an operator.
func PipeOp9[A, B, C, D, E, F, G, H, I, J any](
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
) func(Observable[A]) Observable[J] {
	return func(source Observable[A]) Observable[J] {
		return Pipe9(
			source,
			operator1,
			operator2,
			operator3,
			operator4,
			operator5,
			operator6,
			operator7,
			operator8,
			operator9,
		)
	}
}

// PipeOp10 is similar to Pipe10, but can be used as an operator.
func PipeOp10[A, B, C, D, E, F, G, H, I, J, K any](
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
) func(Observable[A]) Observable[K] {
	return func(source Observable[A]) Observable[K] {
		return Pipe10(
			source,
			operator1,
			operator2,
			operator3,
			operator4,
			operator5,
			operator6,
			operator7,
			operator8,
			operator9,
			operator10,
		)
	}
}

// PipeOp11 is similar to Pipe11, but can be used as an operator.
func PipeOp11[A, B, C, D, E, F, G, H, I, J, K, L any](
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
) func(Observable[A]) Observable[L] {
	return func(source Observable[A]) Observable[L] {
		return Pipe11(
			source,
			operator1,
			operator2,
			operator3,
			operator4,
			operator5,
			operator6,
			operator7,
			operator8,
			operator9,
			operator10,
			operator11,
		)
	}
}

// PipeOp12 is similar to Pipe12, but can be used as an operator.
func PipeOp12[A, B, C, D, E, F, G, H, I, J, K, L, M any](
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
) func(Observable[A]) Observable[M] {
	return func(source Observable[A]) Observable[M] {
		return Pipe12(
			source,
			operator1,
			operator2,
			operator3,
			operator4,
			operator5,
			operator6,
			operator7,
			operator8,
			operator9,
			operator10,
			operator11,
			operator12,
		)
	}
}

// PipeOp13 is similar to Pipe13, but can be used as an operator.
func PipeOp13[A, B, C, D, E, F, G, H, I, J, K, L, M, N any](
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
) func(Observable[A]) Observable[N] {
	return func(source Observable[A]) Observable[N] {
		return Pipe13(
			source,
			operator1,
			operator2,
			operator3,
			operator4,
			operator5,
			operator6,
			operator7,
			operator8,
			operator9,
			operator10,
			operator11,
			operator12,
			operator13,
		)
	}
}

// PipeOp14 is similar to Pipe14, but can be used as an operator.
func PipeOp14[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O any](
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
	operator14 func(Observable[N]) Observable[O],
) func(Observable[A]) Observable[O] {
	return func(source Observable[A]) Observable[O] {
		return Pipe14(
			source,
			operator1,
			operator2,
			operator3,
			operator4,
			operator5,
			operator6,
			operator7,
			operator8,
			operator9,
			operator10,
			operator11,
			operator12,
			operator13,
			operator14,
		)
	}
}

// PipeOp15 is similar to Pipe15, but can be used as an operator.
func PipeOp15[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P any](
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
	operator14 func(Observable[N]) Observable[O],
	operator15 func(Observable[O]) Observable[P],
) func(Observable[A]) Observable[P] {
	return func(source Observable[A]) Observable[P] {
		return Pipe15(
			source,
			operator1,
			operator2,
			operator3,
			operator4,
			operator5,
			operator6,
			operator7,
			operator8,
			operator9,
			operator10,
			operator11,
			operator12,
			operator13,
			operator14,
			operator15,
		)
	}
}

// PipeOp16 is similar to Pipe16, but can be used as an operator.
func PipeOp16[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q any](
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
	operator14 func(Observable[N]) Observable[O],
	operator15 func(Observable[O]) Observable[P],
	operator16 func(Observable[P]) Observable[Q],
) func(Observable[A]) Observable[Q] {
	return func(source Observable[A]) Observable[Q] {
		return Pipe16(
			source,
			operator1,
			operator2,
			operator3,
			operator4,
			operator5,
			operator6,
			operator7,
			operator8,
			operator9,
			operator10,
			operator11,
			operator12,
			operator13,
			operator14,
			operator15,
			operator16,
		)
	}
}

// PipeOp17 is similar to Pipe17, but can be used as an operator.
func PipeOp17[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R any](
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
	operator14 func(Observable[N]) Observable[O],
	operator15 func(Observable[O]) Observable[P],
	operator16 func(Observable[P]) Observable[Q],
	operator17 func(Observable[Q]) Observable[R],
) func(Observable[A]) Observable[R] {
	return func(source Observable[A]) Observable[R] {
		return Pipe17(
			source,
			operator1,
			operator2,
			operator3,
			operator4,
			operator5,
			operator6,
			operator7,
			operator8,
			operator9,
			operator10,
			operator11,
			operator12,
			operator13,
			operator14,
			operator15,
			operator16,
			operator17,
		)
	}
}

// PipeOp18 is similar to Pipe18, but can be used as an operator.
func PipeOp18[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S any](
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
	operator14 func(Observable[N]) Observable[O],
	operator15 func(Observable[O]) Observable[P],
	operator16 func(Observable[P]) Observable[Q],
	operator17 func(Observable[Q]) Observable[R],
	operator18 func(Observable[R]) Observable[S],
) func(Observable[A]) Observable[S] {
	return func(source Observable[A]) Observable[S] {
		return Pipe18(
			source,
			operator1,
			operator2,
			operator3,
			operator4,
			operator5,
			operator6,
			operator7,
			operator8,
			operator9,
			operator10,
			operator11,
			operator12,
			operator13,
			operator14,
			operator15,
			operator16,
			operator17,
			operator18,
		)
	}
}

// PipeOp19 is similar to Pipe19, but can be used as an operator.
func PipeOp19[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T any](
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
	operator14 func(Observable[N]) Observable[O],
	operator15 func(Observable[O]) Observable[P],
	operator16 func(Observable[P]) Observable[Q],
	operator17 func(Observable[Q]) Observable[R],
	operator18 func(Observable[R]) Observable[S],
	operator19 func(Observable[S]) Observable[T],
) func(Observable[A]) Observable[T] {
	return func(source Observable[A]) Observable[T] {
		return Pipe19(
			source,
			operator1,
			operator2,
			operator3,
			operator4,
			operator5,
			operator6,
			operator7,
			operator8,
			operator9,
			operator10,
			operator11,
			operator12,
			operator13,
			operator14,
			operator15,
			operator16,
			operator17,
			operator18,
			operator19,
		)
	}
}

// PipeOp20 is similar to Pipe20, but can be used as an operator.
func PipeOp20[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U any](
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
	operator14 func(Observable[N]) Observable[O],
	operator15 func(Observable[O]) Observable[P],
	operator16 func(Observable[P]) Observable[Q],
	operator17 func(Observable[Q]) Observable[R],
	operator18 func(Observable[R]) Observable[S],
	operator19 func(Observable[S]) Observable[T],
	operator20 func(Observable[T]) Observable[U],
) func(Observable[A]) Observable[U] {
	return func(source Observable[A]) Observable[U] {
		return Pipe20(
			source,
			operator1,
			operator2,
			operator3,
			operator4,
			operator5,
			operator6,
			operator7,
			operator8,
			operator9,
			operator10,
			operator11,
			operator12,
			operator13,
			operator14,
			operator15,
			operator16,
			operator17,
			operator18,
			operator19,
			operator20,
		)
	}
}

// PipeOp21 is similar to Pipe21, but can be used as an operator.
func PipeOp21[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V any](
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
	operator14 func(Observable[N]) Observable[O],
	operator15 func(Observable[O]) Observable[P],
	operator16 func(Observable[P]) Observable[Q],
	operator17 func(Observable[Q]) Observable[R],
	operator18 func(Observable[R]) Observable[S],
	operator19 func(Observable[S]) Observable[T],
	operator20 func(Observable[T]) Observable[U],
	operator21 func(Observable[U]) Observable[V],
) func(Observable[A]) Observable[V] {
	return func(source Observable[A]) Observable[V] {
		return Pipe21(
			source,
			operator1,
			operator2,
			operator3,
			operator4,
			operator5,
			operator6,
			operator7,
			operator8,
			operator9,
			operator10,
			operator11,
			operator12,
			operator13,
			operator14,
			operator15,
			operator16,
			operator17,
			operator18,
			operator19,
			operator20,
			operator21,
		)
	}
}

// PipeOp22 is similar to Pipe22, but can be used as an operator.
func PipeOp22[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W any](
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
	operator14 func(Observable[N]) Observable[O],
	operator15 func(Observable[O]) Observable[P],
	operator16 func(Observable[P]) Observable[Q],
	operator17 func(Observable[Q]) Observable[R],
	operator18 func(Observable[R]) Observable[S],
	operator19 func(Observable[S]) Observable[T],
	operator20 func(Observable[T]) Observable[U],
	operator21 func(Observable[U]) Observable[V],
	operator22 func(Observable[V]) Observable[W],
) func(Observable[A]) Observable[W] {
	return func(source Observable[A]) Observable[W] {
		return Pipe22(
			source,
			operator1,
			operator2,
			operator3,
			operator4,
			operator5,
			operator6,
			operator7,
			operator8,
			operator9,
			operator10,
			operator11,
			operator12,
			operator13,
			operator14,
			operator15,
			operator16,
			operator17,
			operator18,
			operator19,
			operator20,
			operator21,
			operator22,
		)
	}
}

// PipeOp23 is similar to Pipe23, but can be used as an operator.
func PipeOp23[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X any](
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
	operator14 func(Observable[N]) Observable[O],
	operator15 func(Observable[O]) Observable[P],
	operator16 func(Observable[P]) Observable[Q],
	operator17 func(Observable[Q]) Observable[R],
	operator18 func(Observable[R]) Observable[S],
	operator19 func(Observable[S]) Observable[T],
	operator20 func(Observable[T]) Observable[U],
	operator21 func(Observable[U]) Observable[V],
	operator22 func(Observable[V]) Observable[W],
	operator23 func(Observable[W]) Observable[X],
) func(Observable[A]) Observable[X] {
	return func(source Observable[A]) Observable[X] {
		return Pipe23(
			source,
			operator1,
			operator2,
			operator3,
			operator4,
			operator5,
			operator6,
			operator7,
			operator8,
			operator9,
			operator10,
			operator11,
			operator12,
			operator13,
			operator14,
			operator15,
			operator16,
			operator17,
			operator18,
			operator19,
			operator20,
			operator21,
			operator22,
			operator23,
		)
	}
}

// PipeOp24 is similar to Pipe24, but can be used as an operator.
func PipeOp24[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y any](
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
	operator14 func(Observable[N]) Observable[O],
	operator15 func(Observable[O]) Observable[P],
	operator16 func(Observable[P]) Observable[Q],
	operator17 func(Observable[Q]) Observable[R],
	operator18 func(Observable[R]) Observable[S],
	operator19 func(Observable[S]) Observable[T],
	operator20 func(Observable[T]) Observable[U],
	operator21 func(Observable[U]) Observable[V],
	operator22 func(Observable[V]) Observable[W],
	operator23 func(Observable[W]) Observable[X],
	operator24 func(Observable[X]) Observable[Y],
) func(Observable[A]) Observable[Y] {
	return func(source Observable[A]) Observable[Y] {
		return Pipe24(
			source,
			operator1,
			operator2,
			operator3,
			operator4,
			operator5,
			operator6,
			operator7,
			operator8,
			operator9,
			operator10,
			operator11,
			operator12,
			operator13,
			operator14,
			operator15,
			operator16,
			operator17,
			operator18,
			operator19,
			operator20,
			operator21,
			operator22,
			operator23,
			operator24,
		)
	}
}

// PipeOp25 is similar to Pipe25, but can be used as an operator.
func PipeOp25[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z any](
	operator1 func(Observable[A]) Observable[B],
	operator2 func(Observable[B]) Observable[C],
	operator3 func(Observable[C]) Observable[D],
	operator4 func(Observable[D]) Observable[E],
	operator5 func(Observable[E]) Observable[F],
	operator6 func(Observable[F]) Observable[G],
	operator7 func(Observable[G]) Observable[H],
	operator8 func(Observable[H]) Observable[I],
	operator9 func(Observable[I]) Observable[J],
	operator10 func(Observable[J]) Observable[K],
	operator11 func(Observable[K]) Observable[L],
	operator12 func(Observable[L]) Observable[M],
	operator13 func(Observable[M]) Observable[N],
	operator14 func(Observable[N]) Observable[O],
	operator15 func(Observable[O]) Observable[P],
	operator16 func(Observable[P]) Observable[Q],
	operator17 func(Observable[Q]) Observable[R],
	operator18 func(Observable[R]) Observable[S],
	operator19 func(Observable[S]) Observable[T],
	operator20 func(Observable[T]) Observable[U],
	operator21 func(Observable[U]) Observable[V],
	operator22 func(Observable[V]) Observable[W],
	operator23 func(Observable[W]) Observable[X],
	operator24 func(Observable[X]) Observable[Y],
	operator25 func(Observable[Y]) Observable[Z],
) func(Observable[A]) Observable[Z] {
	return func(source Observable[A]) Observable[Z] {
		return Pipe25(
			source,
			operator1,
			operator2,
			operator3,
			operator4,
			operator5,
			operator6,
			operator7,
			operator8,
			operator9,
			operator10,
			operator11,
			operator12,
			operator13,
			operator14,
			operator15,
			operator16,
			operator17,
			operator18,
			operator19,
			operator20,
			operator21,
			operator22,
			operator23,
			operator24,
			operator25,
		)
	}
}
