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

package rowebsocketclient

import "errors"

var (
	ErrWebsocketSubjectURLRequired          = errors.New("rowebsocketclient.NewWebsocketSubject: URL is required")
	ErrWebsocketSubjectSerializerRequired   = errors.New("rowebsocketclient.NewWebsocketSubject: Serializer is required")
	ErrWebsocketSubjectDeserializerRequired = errors.New("rowebsocketclient.NewWebsocketSubject: Deserializer is required")
)
