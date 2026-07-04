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

package rostrings

type caseTests struct {
	PascalCase string
	CamelCase  string
	KebabCase  string
	SnakeCase  string
}

var allCaseTests = []struct {
	input  string
	output caseTests
}{
	{
		output: caseTests{},
	},
	{
		input:  ".",
		output: caseTests{},
	},
	{
		input: "Hello world!",
		output: caseTests{
			PascalCase: "HelloWorld",
			CamelCase:  "helloWorld",
			KebabCase:  "hello-world",
			SnakeCase:  "hello_world",
		},
	},
	{
		input: "A",
		output: caseTests{
			PascalCase: "A",
			CamelCase:  "a",
			KebabCase:  "a",
			SnakeCase:  "a",
		},
	},
	{
		input: "a",
		output: caseTests{
			PascalCase: "A",
			CamelCase:  "a",
			KebabCase:  "a",
			SnakeCase:  "a",
		},
	},
	{
		input: "foo",
		output: caseTests{
			PascalCase: "Foo",
			CamelCase:  "foo",
			KebabCase:  "foo",
			SnakeCase:  "foo",
		},
	},
	{
		input: "snake_case",
		output: caseTests{
			PascalCase: "SnakeCase",
			CamelCase:  "snakeCase",
			KebabCase:  "snake-case",
			SnakeCase:  "snake_case",
		},
	},
	{
		input: "SNAKE_CASE",
		output: caseTests{
			PascalCase: "SnakeCase",
			CamelCase:  "snakeCase",
			KebabCase:  "snake-case",
			SnakeCase:  "snake_case",
		},
	},
	{
		input: "kebab-case",
		output: caseTests{
			PascalCase: "KebabCase",
			CamelCase:  "kebabCase",
			KebabCase:  "kebab-case",
			SnakeCase:  "kebab_case",
		},
	},
	{
		input: "PascalCase",
		output: caseTests{
			PascalCase: "PascalCase",
			CamelCase:  "pascalCase",
			KebabCase:  "pascal-case",
			SnakeCase:  "pascal_case",
		},
	},
	{
		input: "camelCase",
		output: caseTests{
			PascalCase: "CamelCase",
			CamelCase:  "camelCase",
			KebabCase:  `camel-case`,
			SnakeCase:  "camel_case",
		},
	},
	{
		input: "Title Case",
		output: caseTests{
			PascalCase: "TitleCase",
			CamelCase:  "titleCase",
			KebabCase:  "title-case",
			SnakeCase:  "title_case",
		},
	},
	{
		input: "point.case",
		output: caseTests{
			PascalCase: "PointCase",
			CamelCase:  "pointCase",
			KebabCase:  "point-case",
			SnakeCase:  "point_case",
		},
	},
	{
		input: "snake_case_with_more_words",
		output: caseTests{
			PascalCase: "SnakeCaseWithMoreWords",
			CamelCase:  "snakeCaseWithMoreWords",
			KebabCase:  "snake-case-with-more-words",
			SnakeCase:  "snake_case_with_more_words",
		},
	},
	{
		input: "SNAKE_CASE_WITH_MORE_WORDS",
		output: caseTests{
			PascalCase: "SnakeCaseWithMoreWords",
			CamelCase:  "snakeCaseWithMoreWords",
			KebabCase:  "snake-case-with-more-words",
			SnakeCase:  "snake_case_with_more_words",
		},
	},
	{
		input: "kebab-case-with-more-words",
		output: caseTests{
			PascalCase: "KebabCaseWithMoreWords",
			CamelCase:  "kebabCaseWithMoreWords",
			KebabCase:  "kebab-case-with-more-words",
			SnakeCase:  "kebab_case_with_more_words",
		},
	},
	{
		input: "PascalCaseWithMoreWords",
		output: caseTests{
			PascalCase: "PascalCaseWithMoreWords",
			CamelCase:  "pascalCaseWithMoreWords",
			KebabCase:  "pascal-case-with-more-words",
			SnakeCase:  "pascal_case_with_more_words",
		},
	},
	{
		input: "camelCaseWithMoreWords",
		output: caseTests{
			PascalCase: "CamelCaseWithMoreWords",
			CamelCase:  "camelCaseWithMoreWords",
			KebabCase:  "camel-case-with-more-words",
			SnakeCase:  "camel_case_with_more_words",
		},
	},
	{
		input: "Title Case With More Words",
		output: caseTests{
			PascalCase: "TitleCaseWithMoreWords",
			CamelCase:  "titleCaseWithMoreWords",
			KebabCase:  "title-case-with-more-words",
			SnakeCase:  "title_case_with_more_words",
		},
	},
	{
		input: "point.case.with.more.words",
		output: caseTests{
			PascalCase: "PointCaseWithMoreWords",
			CamelCase:  "pointCaseWithMoreWords",
			KebabCase:  "point-case-with-more-words",
			SnakeCase:  "point_case_with_more_words",
		},
	},
	{
		input: "snake_case__with___multiple____delimiters",
		output: caseTests{
			PascalCase: "SnakeCaseWithMultipleDelimiters",
			CamelCase:  "snakeCaseWithMultipleDelimiters",
			KebabCase:  "snake-case-with-multiple-delimiters",
			SnakeCase:  "snake_case_with_multiple_delimiters",
		},
	},
	{
		input: "SNAKE_CASE__WITH___multiple____DELIMITERS",
		output: caseTests{
			PascalCase: "SnakeCaseWithMultipleDelimiters",
			CamelCase:  "snakeCaseWithMultipleDelimiters",
			KebabCase:  "snake-case-with-multiple-delimiters",
			SnakeCase:  "snake_case_with_multiple_delimiters",
		},
	},
	{
		input: "kebab-case--with---multiple----delimiters",
		output: caseTests{
			PascalCase: "KebabCaseWithMultipleDelimiters",
			CamelCase:  "kebabCaseWithMultipleDelimiters",
			KebabCase:  "kebab-case-with-multiple-delimiters",
			SnakeCase:  "kebab_case_with_multiple_delimiters",
		},
	},
	{
		input: "Title Case  With   Multiple    Delimiters",
		output: caseTests{
			PascalCase: "TitleCaseWithMultipleDelimiters",
			CamelCase:  "titleCaseWithMultipleDelimiters",
			KebabCase:  "title-case-with-multiple-delimiters",
			SnakeCase:  "title_case_with_multiple_delimiters",
		},
	},
	{
		input: "point.case..with...multiple....delimiters",
		output: caseTests{
			PascalCase: "PointCaseWithMultipleDelimiters",
			CamelCase:  "pointCaseWithMultipleDelimiters",
			KebabCase:  "point-case-with-multiple-delimiters",
			SnakeCase:  "point_case_with_multiple_delimiters",
		},
	},
	{
		input: " leading space",
		output: caseTests{
			PascalCase: "LeadingSpace",
			CamelCase:  "leadingSpace",
			KebabCase:  "leading-space",
			SnakeCase:  "leading_space",
		},
	},
	{
		input: "   leading spaces",
		output: caseTests{
			PascalCase: "LeadingSpaces",
			CamelCase:  "leadingSpaces",
			KebabCase:  "leading-spaces",
			SnakeCase:  "leading_spaces",
		},
	},
	{
		input: "\t\t\r\n leading whitespaces",
		output: caseTests{
			PascalCase: "LeadingWhitespaces",
			CamelCase:  "leadingWhitespaces",
			KebabCase:  "leading-whitespaces",
			SnakeCase:  "leading_whitespaces",
		},
	},
	{
		input: "trailing space ",
		output: caseTests{
			PascalCase: "TrailingSpace",
			CamelCase:  "trailingSpace",
			KebabCase:  "trailing-space",
			SnakeCase:  "trailing_space",
		},
	},
	{
		input: "trailing spaces   ",
		output: caseTests{
			PascalCase: "TrailingSpaces",
			CamelCase:  "trailingSpaces",
			KebabCase:  "trailing-spaces",
			SnakeCase:  "trailing_spaces",
		},
	},
	{
		input: "trailing whitespaces\t\t\r\n",
		output: caseTests{
			PascalCase: "TrailingWhitespaces",
			CamelCase:  "trailingWhitespaces",
			KebabCase:  "trailing-whitespaces",
			SnakeCase:  "trailing_whitespaces",
		},
	},
	{
		input: " on both sides ",
		output: caseTests{
			PascalCase: "OnBothSides",
			CamelCase:  "onBothSides",
			KebabCase:  "on-both-sides",
			SnakeCase:  "on_both_sides",
		},
	},
	{
		input: "    many on both sides  ",
		output: caseTests{
			PascalCase: "ManyOnBothSides",
			CamelCase:  "manyOnBothSides",
			KebabCase:  "many-on-both-sides",
			SnakeCase:  "many_on_both_sides",
		},
	},
	{
		input: "\r whitespaces on both sides\t\t\r\n",
		output: caseTests{
			PascalCase: "WhitespacesOnBothSides",
			CamelCase:  "whitespacesOnBothSides",
			KebabCase:  "whitespaces-on-both-sides",
			SnakeCase:  "whitespaces_on_both_sides",
		},
	},
	{
		input: "  extraSpaces in_This TestCase Of MIXED_CASES\t",
		output: caseTests{
			PascalCase: "ExtraSpacesInThisTestCaseOfMixedCases",
			CamelCase:  "extraSpacesInThisTestCaseOfMixedCases",
			KebabCase:  "extra-spaces-in-this-test-case-of-mixed-cases",
			SnakeCase:  "extra_spaces_in_this_test_case_of_mixed_cases",
		},
	},
	{
		input: "CASEBreak",
		output: caseTests{
			PascalCase: "CaseBreak",
			CamelCase:  "caseBreak",
			KebabCase:  "case-break",
			SnakeCase:  "case_break",
		},
	},
	{
		input: "ID",
		output: caseTests{
			PascalCase: "Id",
			CamelCase:  "id",
			KebabCase:  "id",
			SnakeCase:  "id",
		},
	},
	{
		input: "userID",
		output: caseTests{
			PascalCase: "UserId",
			CamelCase:  "userId",
			KebabCase:  "user-id",
			SnakeCase:  "user_id",
		},
	},
	{
		input: "JSON_blob",
		output: caseTests{
			PascalCase: "JsonBlob",
			CamelCase:  "jsonBlob",
			KebabCase:  "json-blob",
			SnakeCase:  "json_blob",
		},
	},
	{
		input: "HTTPStatusCode",
		output: caseTests{
			PascalCase: "HttpStatusCode",
			CamelCase:  "httpStatusCode",
			KebabCase:  "http-status-code",
			SnakeCase:  "http_status_code",
		},
	},
	{
		input: "FreeBSD and SSLError are not golang initialisms",
		output: caseTests{
			PascalCase: "FreeBsdAndSslErrorAreNotGolangInitialisms",
			CamelCase:  "freeBsdAndSslErrorAreNotGolangInitialisms",
			KebabCase:  "free-bsd-and-ssl-error-are-not-golang-initialisms",
			SnakeCase:  "free_bsd_and_ssl_error_are_not_golang_initialisms",
		},
	},
	{
		input: "David's Computer",
		output: caseTests{
			PascalCase: "DavidSComputer",
			CamelCase:  "davidSComputer",
			KebabCase:  "david-s-computer",
			SnakeCase:  "david_s_computer",
		},
	},
	{
		input: "http200",
		output: caseTests{
			PascalCase: "Http200",
			CamelCase:  "http200",
			KebabCase:  "http-200",
			SnakeCase:  "http_200",
		},
	},
	{
		input: "NumberSplittingVersion1.0r3",
		output: caseTests{
			PascalCase: "NumberSplittingVersion10R3",
			CamelCase:  "numberSplittingVersion10R3",
			KebabCase:  "number-splitting-version-1-0-r3",
			SnakeCase:  "number_splitting_version_1_0_r3",
		},
	},
	{
		input: "When you have a comma, odd results",
		output: caseTests{
			PascalCase: "WhenYouHaveACommaOddResults",
			CamelCase:  "whenYouHaveACommaOddResults",
			KebabCase:  "when-you-have-a-comma-odd-results",
			SnakeCase:  "when_you_have_a_comma_odd_results",
		},
	},
	{
		input: "Ordinal numbers work: 1st 2nd and 3rd place",
		output: caseTests{
			PascalCase: "OrdinalNumbersWork1St2NdAnd3RdPlace",
			CamelCase:  "ordinalNumbersWork1St2NdAnd3RdPlace",
			KebabCase:  "ordinal-numbers-work-1-st-2-nd-and-3-rd-place",
			SnakeCase:  "ordinal_numbers_work_1_st_2_nd_and_3_rd_place",
		},
	},
	{
		input: "BadUTF8\xe2\xe2\xa1",
		output: caseTests{
			PascalCase: "BadUtf8",
			CamelCase:  "badUtf8",
			KebabCase:  "bad-utf-8",
			SnakeCase:  "bad_utf_8",
		},
	},
	{
		input: "IDENT3",
		output: caseTests{
			PascalCase: "Ident3",
			CamelCase:  "ident3",
			KebabCase:  "ident-3",
			SnakeCase:  "ident_3",
		},
	},
	{
		input: "LogRouterS3BucketName",
		output: caseTests{
			PascalCase: "LogRouterS3BucketName",
			CamelCase:  "logRouterS3BucketName",
			KebabCase:  "log-router-s3-bucket-name",
			SnakeCase:  "log_router_s3_bucket_name",
		},
	},
	{
		input: "PINEAPPLE",
		output: caseTests{
			PascalCase: "Pineapple",
			CamelCase:  "pineapple",
			KebabCase:  "pineapple",
			SnakeCase:  "pineapple",
		},
	},
	{
		input: "Int8Value",
		output: caseTests{
			PascalCase: "Int8Value",
			CamelCase:  "int8Value",
			KebabCase:  "int-8-value",
			SnakeCase:  "int_8_value",
		},
	},
	{
		input: "first.last",
		output: caseTests{
			PascalCase: "FirstLast",
			CamelCase:  "firstLast",
			KebabCase:  "first-last",
			SnakeCase:  "first_last",
		},
	},
}
