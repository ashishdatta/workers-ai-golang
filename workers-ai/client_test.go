package workersai

// nolint:errcheck
import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestClient_Chat(t *testing.T) {
	// The mock response must now be a valid JSON that ChatResponse.UnmarshalJSON can parse.
	// This is a "legacy" style response for a simple chat message.
	mockResponseJSON := `{
		"success": true,
		"errors": [],
		"messages": [],
		"result": {
			"response": "Hello! How can I help you today?",
			"usage": {
				"prompt_tokens": 15,
				"completion_tokens": 9,
				"total_tokens": 24
			}
		}
	}`

	// Set up the mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "POST", r.Method, "Expected POST request")
		require.Equal(t, "Bearer test-token", r.Header.Get("Authorization"), "Expected correct Authorization header")
		require.Equal(t, "application/json", r.Header.Get("Content-Type"), "Expected correct Content-Type header")

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, err := w.Write([]byte(mockResponseJSON))
		require.NoError(t, err)
	}))
	defer server.Close()

	// Initialize the client to use the mock server
	client := NewClient("test-account", "test-token")
	client.BaseURL = server.URL

	// Correctly initialize the messages slice with the concrete type 'ChatMessage'
	messages := []Message{
		ChatMessage{Role: "system", Content: "You are a helpful assistant"},
		ChatMessage{Role: "user", Content: "Hello"},
	}

	// Call the function under test
	response, err := client.Chat("@cf/meta/llama-2-7b-chat-int8", messages, nil)

	// Assert the results
	require.NoError(t, err)
	require.NotNil(t, response)
	require.True(t, response.Success)
	require.True(t, response.IsLegacyResult, "Should detect legacy format for simple chat")

	// Use the GetContent() helper to abstract away the response format
	expectedContent := "Hello! How can I help you today?"
	require.Equal(t, expectedContent, response.GetContent())

	// You can also check the specific legacy field if needed
	require.Equal(t, expectedContent, response.LegacyResponse.Response)
}

func TestClient_Chat_WithModelParameters(t *testing.T) {
	mockResponseJSON := `{
		"success": true,
		"errors": [],
		"messages": [],
		"result": {
			"response": "Hello! How can I help you today?",
			"usage": {
				"prompt_tokens": 15,
				"completion_tokens": 9,
				"total_tokens": 24
			}
		}
	}`

	wantMaxTokens := int64(10)
	wantTopK := 2
	wantTopP := 0.2
	wantTemp := 0.23

	// Set up the mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "POST", r.Method, "Expected POST request")
		require.Equal(t, "Bearer test-token", r.Header.Get("Authorization"), "Expected correct Authorization header")
		require.Equal(t, "application/json", r.Header.Get("Content-Type"), "Expected correct Content-Type header")

		b, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal(err)
		}

		req := &ChatCompletionRequest{}
		if err := json.Unmarshal(b, req); err != nil {
			t.Fatal(err)
		}

		assert.Equal(t, req.MaxTokens, wantMaxTokens)
		assert.Equal(t, req.TopK, wantTopK)
		assert.Equal(t, req.TopP, wantTopP)
		assert.Equal(t, req.Temperature, wantTemp)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, err = w.Write([]byte(mockResponseJSON))
		require.NoError(t, err)
	}))
	defer server.Close()

	// Initialize the client to use the mock server
	client := NewClient("test-account", "test-token")
	client.BaseURL = server.URL

	// Correctly initialize the messages slice with the concrete type 'ChatMessage'
	messages := []Message{
		ChatMessage{Role: "system", Content: "You are a helpful assistant"},
		ChatMessage{Role: "user", Content: "Hello"},
	}

	mp := &ModelParameters{
		MaxTokens:   int64(wantMaxTokens),
		TopK:        wantTopK,
		TopP:        wantTopP,
		Temperature: wantTemp,
	}

	// Call the function under test
	_, gotErr := client.Chat("@cf/meta/llama-2-7b-chat-int8", messages, mp)
	assert.Nil(t, gotErr)
}

func TestClient_Chat_WithoutModelParameters(t *testing.T) {
	mockResponseJSON := `{
		"success": true,
		"errors": [],
		"messages": [],
		"result": {
			"response": "Hello! How can I help you today?",
			"usage": {
				"prompt_tokens": 15,
				"completion_tokens": 9,
				"total_tokens": 24
			}
		}
	}`

	// Set up the mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "POST", r.Method, "Expected POST request")
		require.Equal(t, "Bearer test-token", r.Header.Get("Authorization"), "Expected correct Authorization header")
		require.Equal(t, "application/json", r.Header.Get("Content-Type"), "Expected correct Content-Type header")

		b, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal(err)
		}

		req := &ChatCompletionRequest{}
		if err := json.Unmarshal(b, req); err != nil {
			t.Fatal(err)
		}

		assert.Zero(t, req.MaxTokens)
		assert.Zero(t, req.TopK)
		assert.Zero(t, req.TopP)
		assert.Zero(t, req.Temperature)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, err = w.Write([]byte(mockResponseJSON))
		require.NoError(t, err)
	}))
	defer server.Close()

	// Initialize the client to use the mock server
	client := NewClient("test-account", "test-token")
	client.BaseURL = server.URL

	// Correctly initialize the messages slice with the concrete type 'ChatMessage'
	messages := []Message{
		ChatMessage{Role: "system", Content: "You are a helpful assistant"},
		ChatMessage{Role: "user", Content: "Hello"},
	}

	_, gotErr := client.Chat("@cf/meta/llama-2-7b-chat-int8", messages, nil)
	assert.Nil(t, gotErr)
}

func TestClient_GetModelInfo(t *testing.T) {
	mockResponse := ModelInfo{
		Name:        "Test Model",
		Description: "A test model for unit testing",
		Task: struct {
			Name        string `json:"name"`
			Description string `json:"description"`
		}{
			Name:        "text-generation",
			Description: "Generates text based on input",
		},
		Properties: struct {
			MaxBatchSize   int `json:"max_batch_size"`
			MaxTotalTokens int `json:"max_total_tokens"`
		}{
			MaxBatchSize:   1,
			MaxTotalTokens: 4096,
		},
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" {
			t.Errorf("Expected GET request, got %s", r.Method)
		}

		if r.Header.Get("Authorization") != "Bearer test-token" {
			t.Errorf("Expected Authorization header with Bearer token")
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(mockResponse)
	}))
	defer server.Close()

	client := NewClient("test-account", "test-token")
	client.BaseURL = server.URL

	modelInfo, err := client.GetModelInfo(ModelLlama4Scout17B)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if modelInfo.Name != "Test Model" {
		t.Errorf("Expected model name 'Test Model', got %s", modelInfo.Name)
	}

	if modelInfo.Properties.MaxTotalTokens != 4096 {
		t.Errorf("Expected max tokens 4096, got %d", modelInfo.Properties.MaxTotalTokens)
	}
}

// TODO: fix - the method currently queries upstream directly, requires a test mode
//func TestClient_ListModels(t *testing.T) {
//	mockResponse := ModelsResponse{
//		"@cf/meta/llama-3-8b-instruct": &ModelInfo{
//			Description: "First test model",
//			Task: struct {
//				Name        string `json:"name"`
//				Description string `json:"description"`
//			}{
//				Name:        "text-generation",
//				Description: "Generates text",
//			},
//			Parameters: map[string]*Parameter{
//				"max_tokens": {
//					Type:        "integer",
//					Description: "Maximum tokens",
//					Default:     256,
//				},
//			},
//		},
//		"@cf/meta/llama-3-70b-instruct": &ModelInfo{
//			Description: "Second test model",
//			Task: struct {
//				Name        string `json:"name"`
//				Description string `json:"description"`
//			}{
//				Name:        "text-generation",
//				Description: "Generates text",
//			},
//			Parameters: map[string]*Parameter{
//				"temperature": {
//					Type:        "number",
//					Description: "Controls randomness",
//					Default:     0.15,
//				},
//			},
//		},
//	}
//
//	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
//		if r.Method != "GET" {
//			t.Errorf("Expected GET request, got %s", r.Method)
//		}
//
//		w.Header().Set("Content-Type", "application/json")
//		json.NewEncoder(w).Encode(mockResponse)
//	}))
//	defer server.Close()
//
//	client := NewClient("test-account", "test-token")
//	client.BaseURL = server.URL
//
//	models, err := client.ListModels()
//	if err != nil {
//		t.Fatalf("Expected no error, got %v", err)
//	}
//
//	if len(models) != 2 {
//		t.Errorf("Expected 2 models, got %d", len(models))
//	}
//
//	// Check that model names are set from map keys
//	modelNames := make(map[string]bool)
//	for _, model := range models {
//		modelNames[model.Name] = true
//	}
//
//	if !modelNames["@cf/meta/llama-3-8b-instruct"] {
//		t.Error("Expected model '@cf/meta/llama-3-8b-instruct' not found")
//	}
//
//	if !modelNames["@cf/meta/llama-3-70b-instruct"] {
//		t.Error("Expected model '@cf/meta/llama-3-70b-instruct' not found")
//	}
//}

func TestClient_Chat_Integration(t *testing.T) {
	accountID := os.Getenv("CLOUDFLARE_ACCOUNT_ID")
	apiToken := os.Getenv("CLOUDFLARE_AUTH_TOKEN")

	if accountID == "" || apiToken == "" {
		t.Skip("Skipping integration test: CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_AUTH_TOKEN not set")
	}

	client := NewClient(accountID, apiToken)

	messages := []Message{
		ChatMessage{Role: "system", Content: "You are a helpful assistant. Keep responses brief."},
		ChatMessage{Role: "user", Content: "Say 'Hello World' and nothing else."},
	}

	response, err := client.Chat(ModelLlama4Scout17B, messages, nil)
	if err != nil {
		t.Fatalf("Integration test failed: %v", err)
	}

	if response.LegacyResponse.Response == "" {
		t.Error("Expected non-empty response")
	}

	t.Logf("AI Response: %s", response.LegacyResponse.Response)
}

func TestClient_ListModels_Integration(t *testing.T) {
	accountID := os.Getenv("CLOUDFLARE_ACCOUNT_ID")
	apiToken := os.Getenv("CLOUDFLARE_AUTH_TOKEN")

	if accountID == "" || apiToken == "" {
		t.Skip("Skipping integration test: CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_AUTH_TOKEN not set")
	}

	client := NewClient(accountID, apiToken)

	models, err := client.ListModels()
	if err != nil {
		t.Fatalf("Integration test failed: %v", err)
	}

	if len(models) == 0 {
		t.Error("Expected at least one model")
	}

	t.Logf("Found %d models", len(models))
	for i, model := range models {
		if i >= 3 {
			break
		}
		t.Logf("Model %d: %s - %s", i+1, model.Name, model.Description)
	}
}

// TestChatWithTools_OpenAIResponse_ToolCall tests the happy path for the modern,
// OpenAI-compatible response format where the API returns a tool call.
func TestChatWithTools_OpenAIResponse_ToolCall(t *testing.T) {
	// 1. Setup a mock HTTP server
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Assert that the request is what we expect
		assert.Equal(t, "POST", r.Method)
		assert.Contains(t, r.URL.Path, "/ai/run/@cf/test-model")
		assert.Equal(t, "Bearer test-token", r.Header.Get("Authorization"))

		var reqBody ChatCompletionRequest
		b, err := io.ReadAll(r.Body)
		assert.Nil(t, err)

		err = json.NewDecoder(bytes.NewBuffer(b)).Decode(&reqBody)
		assert.NoError(t, err)
		assert.Len(t, reqBody.Tools, 1)
		assert.Equal(t, "get_weather", reqBody.Tools[0].Function.Name)

		// 2. Define the mock API response in the standard OpenAI format
		mockResponse := `{
			"success": true,
			"errors": [],
			"messages": [],
			"result": {
				"id": "chatcmpl-mock-id",
				"model": "@cf/test-model",
				"object": "chat.completion",
				"created": 1721997900,
				"choices": [
					{
						"finish_reason": "tool_calls",
						"index": 0,
						"message": {
							"role": "assistant",
							"tool_calls": [
								{
									"id": "call_abc123",
									"type": "function",
									"function": {
										"name": "get_weather",
										"arguments": "{\"location\":\"Eindhoven, NL\"}"
									}
								}
							]
						}
					}
				],
				"usage": {
					"prompt_tokens": 54,
					"completion_tokens": 19,
					"total_tokens": 73
				}
			}
		}`

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(mockResponse))
	}))
	defer mockServer.Close()

	// 3. Instantiate the client pointing to the mock server
	client := NewClient("test-account", "test-token")
	client.BaseURL = mockServer.URL

	// 4. Define test inputs using the new structs
	messages := []Message{
		ChatMessage{Role: "user", Content: "What is the weather in Eindhoven?"},
	}
	tools := []Tool{
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "get_weather",
				Description: "Get the current weather",
				Parameters: FunctionParameters{
					Type:     "object",
					Required: []string{"location"},
					Properties: map[string]*Parameter{
						"location": {Type: "string"},
					},
				},
			},
		},
	}

	// 5. Call the method under test
	response, err := client.ChatWithTools("test-model", messages, tools, nil)

	// 6. Assert the results
	assert.NoError(t, err)
	assert.NotNil(t, response)
	assert.True(t, response.Success)
	assert.False(t, response.IsLegacyResult, "Should detect modern OpenAI format")

	// Assert usage data
	assert.Equal(t, 54, response.ChatCompletionResponse.Usage.PromptTokens)
	assert.Equal(t, 19, response.ChatCompletionResponse.Usage.CompletionTokens)

	// Assert tool calls
	toolCalls := response.GetToolCalls()
	assert.Len(t, toolCalls, 1)
	assert.Equal(t, "call_abc123", toolCalls[0].ID)
	assert.Equal(t, "get_weather", toolCalls[0].Function.Name)
	assert.Equal(t, `{"location":"Eindhoven, NL"}`, toolCalls[0].Function.Arguments)
}

// TestChatWithTools_LegacyTextResponse tests the successful handling of the alternate
// legacy text response format.
func TestChatWithTools_LegacyTextResponse(t *testing.T) {
	expectedContent := "In Portuguese, pineapple is translated as abacaxi."
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mockResponse := fmt.Sprintf(`{
			"success": true,
			"errors": [],
			"messages": [],
			"result": {
				"response": "%s",
				"tool_calls": [],
				"usage": {
					"prompt_tokens": 28,
					"completion_tokens": 17,
					"total_tokens": 45
				}
			}
		}`, expectedContent)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(mockResponse))
	}))
	defer mockServer.Close()

	client := NewClient("test-account", "test-token")
	client.BaseURL = mockServer.URL
	client.Debug = true

	// Define test inputs using the new message format
	messages := []Message{
		ChatMessage{Role: "user", Content: "How do you say pineapple in Portuguese?"},
	}

	response, err := client.ChatWithTools("@cf/test-model", messages, nil, nil)
	assert.NoError(t, err)
	assert.NotNil(t, response)
	assert.True(t, response.Success)
	assert.True(t, response.IsLegacyResult, "Should detect legacy format")

	assert.Equal(t, expectedContent, response.GetContent())
	assert.Empty(t, response.GetToolCalls())

	// Assert usage data from the legacy response
	assert.Equal(t, 28, response.LegacyResponse.Usage.PromptTokens)
	assert.Equal(t, 17, response.LegacyResponse.Usage.CompletionTokens)
	assert.Equal(t, 45, response.LegacyResponse.Usage.TotalTokens)
}

// TestChatWithTools_LegacyToolCallResponse tests the successful handling of the legacy
// tool call format.
func TestChatWithTools_LegacyToolCallResponse(t *testing.T) {
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Define the mock API response in the "legacy" format
		// Note: The arguments are a nested object, not a string like in the OpenAI spec.
		mockResponse := `{
			"success": true,
			"errors": [],
			"messages": [],
			"result": {
				"tool_calls": [
					{
						"name": "gablorken",
						"arguments": {
							"Over": 3.5,
							"Value": 2
						}
					}
				],
				"usage": {
					"prompt_tokens": 135,
					"completion_tokens": 30,
					"total_tokens": 165
				}
			}
		}`
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(mockResponse))
	}))
	defer mockServer.Close()

	client := NewClient("test-account", "test-token")
	client.BaseURL = mockServer.URL

	messages := []Message{
		ChatMessage{Role: "user", Content: "what is a gablorken of 2 over 3.5?"},
	}
	tools := []Tool{
		{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "gablorken",
				Description: "Calculates a gablorken.",
				Parameters: FunctionParameters{
					Type:     "object",
					Required: []string{"Value", "Over"},
					Properties: map[string]*Parameter{
						"Value": {Type: "integer"},
						"Over":  {Type: "number"},
					},
				},
			},
		},
	}

	response, err := client.ChatWithTools("@cf/test-model", messages, tools, nil)
	assert.NoError(t, err)
	assert.NotNil(t, response)
	assert.True(t, response.Success)
	assert.True(t, response.IsLegacyResult, "Should detect legacy format")
	assert.Empty(t, response.GetContent())

	// The legacy response has a different structure for tool calls.
	// We assert against the fields in the LegacyResponse struct.
	legacyToolCalls := response.LegacyResponse.ToolCalls
	assert.Len(t, legacyToolCalls, 1)

	// Note: The legacy ToolCall struct is simpler. It might not have ID or Type.
	// We check the fields that are present in the legacy response.
	// Let's assume the legacy ToolCall struct has Name and Arguments.
	// The `GetToolCalls` accessor should adapt this to the standard `ToolCall` struct.
	adaptedToolCalls := response.GetToolCalls()
	assert.Len(t, adaptedToolCalls, 1)
	assert.Equal(t, "gablorken", adaptedToolCalls[0].Function.Name)

	// The legacy arguments are a JSON object, but the standard `FunctionToCall` expects a string.
	// The accessor should handle this conversion.
	assert.JSONEq(t, `{"Over":3.5,"Value":2}`, adaptedToolCalls[0].Function.Arguments)

	assert.Equal(t, 135, response.LegacyResponse.Usage.PromptTokens)
	assert.Equal(t, 30, response.LegacyResponse.Usage.CompletionTokens)
}
