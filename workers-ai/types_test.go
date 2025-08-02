package workersai

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestChatResponse_UnmarshalJSON to test the custom unmarshaler against all known API response formats.
func TestChatResponse_UnmarshalJSON(t *testing.T) {
	testCases := []struct {
		name               string
		inputJSON          string
		expectLegacy       bool
		expectedContent    string
		expectedToolCallID string
		expectedToolName   string
		expectError        bool
	}{
		{
			name: "should correctly parse standard OpenAI format with tool calls",
			inputJSON: `{
				"success": true,
				"result": {
					"choices": [{
						"finish_reason": "tool_calls",
						"message": {
							"role": "assistant",
							"tool_calls": [{
								"id": "call_openai_123",
								"type": "function",
								"function": {"name": "get_weather", "arguments": "{}"}
							}]
						}
					}]
				}
			}`,
			expectLegacy:       false,
			expectedToolCallID: "call_openai_123",
			expectedToolName:   "get_weather",
		},
		{
			name: "should correctly parse standard OpenAI format with text response",
			inputJSON: `{
				"success": true,
				"result": {
					"choices": [{
						"finish_reason": "stop",
						"message": { "role": "assistant", "content": "Hello there!" }
					}]
				}
			}`,
			expectLegacy:    false,
			expectedContent: "Hello there!",
		},
		{
			name: "should correctly parse hybrid format with tool calls",
			inputJSON: `{
				"success": true,
				"result": {
					"tool_calls": [{
						"id": "call_hybrid_456",
						"type": "function",
						"function": {"name": "gablorken", "arguments": "{}"}
					}]
				}
			}`,
			expectLegacy:       false,
			expectedToolCallID: "call_hybrid_456",
			expectedToolName:   "gablorken",
		},
		{
			name: "should correctly parse legacy format with tool calls",
			inputJSON: `{
				"success": true,
				"result": {
					"tool_calls": [{
						"name": "legacy_gablorken",
						"arguments": {}
					}]
				}
			}`,
			expectLegacy:     true,
			expectedToolName: "legacy_gablorken",
			// Legacy format has no tool call ID, so we expect the adapted one.
			expectedToolCallID: "legacy-tool-call-0",
		},
		{
			name: "should correctly parse legacy format with text response",
			inputJSON: `{
				"success": true,
				"result": {
					"response": "This is a legacy response."
				}
			}`,
			expectLegacy:    true,
			expectedContent: "This is a legacy response.",
		},
		{
			name: "should handle empty result object",
			inputJSON: `{
				"success": true,
				"result": {}
			}`,
			expectLegacy:    true, // The fallback is legacy
			expectedContent: "",
		},
		{
			name:        "should handle malformed top-level JSON",
			inputJSON:   `{"success": true, "result": {`,
			expectError: true,
		},
		{
			name:        "should handle response in JSON object",
			inputJSON:   `{"result":{"response":{"server_id":"foobar","hello":"world"},"tool_calls":[],"usage":{"prompt_tokens":15061,"completion_tokens":192,"total_tokens":15253}},"success":true,"errors":[],"messages":[]}`,
			expectError: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Act
			var response ChatResponse
			err := json.Unmarshal([]byte(tc.inputJSON), &response)

			// Assert
			if tc.expectError {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			assert.True(t, response.Success)
			assert.Equal(t, tc.expectLegacy, response.IsLegacyResult)

			if tc.expectedContent != "" {
				assert.Equal(t, tc.expectedContent, response.GetContent())
			}

			if tc.expectedToolName != "" {
				toolCalls := response.GetToolCalls()
				require.Len(t, toolCalls, 1)
				assert.Equal(t, tc.expectedToolName, toolCalls[0].Function.Name)
				assert.Equal(t, tc.expectedToolCallID, toolCalls[0].ID)
			}
		})
	}
}

func TestChatCompletionRequest_UnmarshalJSON(t *testing.T) {
	testCases := []struct {
		name           string
		inputJSON      string
		expected       ChatCompletionRequest
		expectErr      bool
		expectedErrMsg string
	}{
		{
			name: "Simple request with user message",
			inputJSON: `{
				"model": "test-model",
				"messages": [{"role": "user", "content": "Hello"}]
			}`,
			expected: ChatCompletionRequest{
				Model: "test-model",
				Messages: []Message{
					ChatMessage{Role: "user", Content: "Hello"},
				},
			},
			expectErr: false,
		},
		{
			name: "Request with user and system messages",
			inputJSON: `{
				"model": "test-model",
				"messages": [
					{"role": "system", "content": "You are an assistant."},
					{"role": "user", "content": "Hi there."}
				]
			}`,
			expected: ChatCompletionRequest{
				Model: "test-model",
				Messages: []Message{
					ChatMessage{Role: "system", Content: "You are an assistant."},
					ChatMessage{Role: "user", Content: "Hi there."},
				},
			},
			expectErr: false,
		},
		{
			name: "Request with assistant text message",
			inputJSON: `{
				"model": "test-model",
				"messages": [
					{"role": "user", "content": "What is Go?"},
					{"role": "assistant", "content": "Go is a language."}
				]
			}`,
			expected: ChatCompletionRequest{
				Model: "test-model",
				Messages: []Message{
					ChatMessage{Role: "user", Content: "What is Go?"},
					ChatMessage{Role: "assistant", Content: "Go is a language."},
				},
			},
			expectErr: false,
		},
		{
			name: "Request with assistant tool call",
			inputJSON: `{
				"model": "test-model",
				"messages": [
					{"role": "user", "content": "Call a tool."},
					{"role": "assistant", "tool_calls": [
						{"id": "tool-123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\":\"Eindhoven\"}"}}
					]}
				]
			}`,
			expected: ChatCompletionRequest{
				Model: "test-model",
				Messages: []Message{
					ChatMessage{Role: "user", Content: "Call a tool."},
					ResponseMessage{
						Role: "assistant",
						ToolCalls: []ToolCall{
							{ID: "tool-123", Type: "function", Function: FunctionToCall{Name: "get_weather", Arguments: "{\"location\":\"Eindhoven\"}"}},
						},
					},
				},
			},
			expectErr: false,
		},
		{
			name: "Request with tool response message",
			inputJSON: `{
				"model": "test-model",
				"messages": [
					{"role": "tool", "tool_call_id": "tool-123", "content": "{\"temp\": 20}"}
				]
			}`,
			expected: ChatCompletionRequest{
				Model: "test-model",
				Messages: []Message{
					ToolMessage{Role: "tool", ToolCallID: "tool-123", Content: "{\"temp\": 20}"},
				},
			},
			expectErr: false,
		},
		{
			name: "Full conversation history",
			inputJSON: `{
				"model": "test-model",
				"messages": [
					{"role": "user", "content": "What is the weather in Eindhoven?"},
					{"role": "assistant", "tool_calls": [
						{"id": "tool-abc", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\":\"Eindhoven\"}"}}
					]},
					{"role": "tool", "tool_call_id": "tool-abc", "content": "{\"temp\": 22}"}
				]
			}`,
			expected: ChatCompletionRequest{
				Model: "test-model",
				Messages: []Message{
					ChatMessage{Role: "user", Content: "What is the weather in Eindhoven?"},
					ResponseMessage{
						Role:      "assistant",
						ToolCalls: []ToolCall{{ID: "tool-abc", Type: "function", Function: FunctionToCall{Name: "get_weather", Arguments: "{\"location\":\"Eindhoven\"}"}}},
					},
					ToolMessage{Role: "tool", ToolCallID: "tool-abc", Content: "{\"temp\": 22}"},
				},
			},
			expectErr: false,
		},
		{
			name: "Invalid JSON input",
			inputJSON: `{
				"model": "test-model",
				"messages": [
			}`,
			expectErr:      true,
			expectedErrMsg: "looking for beginning of value",
		},
		{
			name: "Message with unknown role",
			inputJSON: `{
				"model": "test-model",
				"messages": [
					{"role": "developer", "content": "This should fail."}
				]
			}`,
			expectErr:      true,
			expectedErrMsg: "unknown message role found: developer",
		},
		{
			name: "Message missing role",
			inputJSON: `{
				"model": "test-model",
				"messages": [
					{"content": "This should also fail."}
				]
			}`,
			expectErr:      true,
			expectedErrMsg: "unknown message role found: ",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			r := require.New(t)

			var req ChatCompletionRequest
			err := json.Unmarshal([]byte(tc.inputJSON), &req)

			if tc.expectErr {
				r.Error(err)
				if tc.expectedErrMsg != "" {
					r.ErrorContains(err, tc.expectedErrMsg)
				}
			} else {
				r.NoError(err)
				r.Equal(tc.expected.Model, req.Model)
				r.Equal(len(tc.expected.Messages), len(req.Messages))
				// DeepEqual is suitable here because we're comparing concrete types
				r.Equal(tc.expected.Messages, req.Messages)
			}
		})
	}
}

func TestLegacyResponse_UnmarshalJSON(t *testing.T) {
	testCases := []struct {
		name           string
		inputJSON      string
		expected       LegacyResponse
		expectErr      bool
		expectedErrMsg string
	}{
		{
			name: "should handle response as a simple string",
			inputJSON: `{
				"response": "This is a simple text response.",
				"tool_calls": [],
				"usage": {"prompt_tokens": 10, "completion_tokens": 5}
			}`,
			expected: LegacyResponse{
				Response:  "This is a simple text response.",
				ToolCalls: []LegacyToolCall{},
				Usage:     Usage{PromptTokens: 10, CompletionTokens: 5},
			},
			expectErr: false,
		},
		{
			name: "should handle response as a JSON object",
			inputJSON: `{
				"response": {"server_id": "foobar", "hello": "world"},
				"tool_calls": [],
				"usage": {"prompt_tokens": 20, "completion_tokens": 15}
			}`,
			expected: LegacyResponse{
				Response:  `{"server_id": "foobar", "hello": "world"}`,
				ToolCalls: []LegacyToolCall{},
				Usage:     Usage{PromptTokens: 20, CompletionTokens: 15},
			},
			expectErr: false,
		},
		{
			name: "should handle null response field",
			inputJSON: `{
				"response": null,
				"tool_calls": [],
				"usage": {"prompt_tokens": 5, "completion_tokens": 5}
			}`,
			expected: LegacyResponse{
				Response:  "null",
				ToolCalls: []LegacyToolCall{},
				Usage:     Usage{PromptTokens: 5, CompletionTokens: 5},
			},
			expectErr: false,
		},
		{
			name:           "should return error on invalid JSON",
			inputJSON:      `{"response": "hello",`,
			expectErr:      true,
			expectedErrMsg: "unexpected end of JSON input",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			r := require.New(t)

			var resp LegacyResponse
			err := json.Unmarshal([]byte(tc.inputJSON), &resp)

			if tc.expectErr {
				r.Error(err)
				if tc.expectedErrMsg != "" {
					r.ErrorContains(err, tc.expectedErrMsg)
				}
			} else {
				r.NoError(err)
				r.Equal(tc.expected, resp)
			}
		})
	}
}
