package workersai

import (
	"encoding/json"
	"fmt"
)

// https://platform.openai.com/docs/guides/function-calling?api-mode=responses#overview

// =================================================================================
// Structs for DEFINING Tools (Client -> Server)
// =================================================================================

// Tool represents a single tool that can be provided to the model.
type Tool struct {
	Type     string             `json:"type"` // Always "function".
	Function FunctionDefinition `json:"function"`
}

// FunctionDefinition describes the structure of a function, including its
// name, purpose, and the parameters it accepts.
type FunctionDefinition struct {
	Name        string             `json:"name"`
	Description string             `json:"description,omitempty"`
	Parameters  FunctionParameters `json:"parameters"`
}

// FunctionParameters defines the JSON schema for the arguments of a function.
type FunctionParameters struct {
	Type       string                `json:"type"` // Always "object".
	Properties map[string]*Parameter `json:"properties"`
	Required   []string              `json:"required,omitempty"`
}

// Parameter describes a single property within the function's parameters.
// It aligns with basic JSON schema properties.
type Parameter struct {
	Type        string      `json:"type"`
	Description string      `json:"description,omitempty"`
	Default     interface{} `json:"default,omitempty"`
	Minimum     interface{} `json:"minimum,omitempty"`
	Maximum     interface{} `json:"maximum,omitempty"`
	Enum        []string    `json:"enum,omitempty"`
	Items       *Parameter  `json:"items,omitempty"` // Used when type is "array".
}

// =================================================================================
// Structs for the Chat Conversation Flow
// These represent the messages exchanged between the client and the server.
// =================================================================================

// ChatMessage represents a standard message from a user or an assistant.
// This is used when sending messages to the API.
type ChatMessage struct {
	Role    string `json:"role"`              // "user" or "assistant" or "system".
	Content string `json:"content,omitempty"` // Not used if tool_calls is present.
	// ToolCalls is populated by the model when it decides to call a function.
	// This field should be empty for messages you send, unless you are re-sending
	// the assistant's request for context.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// Implements the marker function that identifies it as a chat message
func (ChatMessage) isMessage() {}

// ToolMessage is a message with the `role` set to "tool", containing the result
// of a function call. This is sent from your client back to the model.
type ToolMessage struct {
	Role       string `json:"role"`         // Always "tool".
	Content    string `json:"content"`      // The return value of the function.
	ToolCallID string `json:"tool_call_id"` // The ID from the ToolCall object you received.
}

// Implements the marker function that identifies it as a chat message
func (ToolMessage) isMessage() {}

// =================================================================================
// Structs for RECEIVING Tool Calls (Server -> Client)
// These are used to parse the model's response when it decides to call a function.
// =================================================================================

// ToolCall represents the model's request to execute a specific tool.
// This object is found in the `tool_calls` array of a ResponseMessage.
type ToolCall struct {
	ID       string         `json:"id"`   // The unique ID for this tool call.
	Type     string         `json:"type"` // Always "function".
	Function FunctionToCall `json:"function"`
}

// FunctionToCall contains the name of the function to be executed and the
// arguments provided by the model.
type FunctionToCall struct {
	Name string `json:"name"`
	// Arguments is a string containing a JSON object with the function's arguments.
	// You will need to unmarshal this string to access the individual values.
	Arguments string `json:"arguments"`
}

// =================================================================================
// Top-Level Request and Response Structs
// These are the main objects for an API interaction.
// =================================================================================

// ChatResponse is the primary response struct. It uses a custom UnmarshalJSON
// method to act as an adapter, parsing different API response formats into a
// consistent structure.
type ChatResponse struct {
	Success   bool            `json:"success"`
	Errors    []string        `json:"errors"`
	Messages  []interface{}   `json:"messages"`
	ResultRaw json.RawMessage `json:"result"`

	// IsLegacyResult is a flag set during unmarshaling to indicate which
	// format was detected.
	IsLegacyResult bool `json:"-"`
	// ChatCompletionResponse holds the standard OpenAI-compatible response.
	ChatCompletionResponse ChatCompletionResponse
	// LegacyResponse holds the legacy response.
	LegacyResponse LegacyResponse
}

// UnmarshalJSON implements the json.Unmarshaler interface for ChatResponse.
// This distinguishes between three cases:
// - standard OpenAI format (with a "choices" array)
// - the hybrid format (modern tool calls without "choices")
// - and the legacy format.
func (cr *ChatResponse) UnmarshalJSON(data []byte) error {
	// A temporary struct is used to capture the top-level fields and the
	// raw, unparsed 'result' JSON.
	type TempChatResponse struct {
		Success   bool            `json:"success"`
		Errors    []string        `json:"errors"`
		Messages  []interface{}   `json:"messages"`
		ResultRaw json.RawMessage `json:"result"`
	}

	var temp TempChatResponse
	if err := json.Unmarshal(data, &temp); err != nil {
		return fmt.Errorf("failed to unmarshal initial response shell: %w", err)
	}

	// Copy the definite top-level fields.
	cr.Success = temp.Success
	cr.Errors = temp.Errors
	cr.Messages = temp.Messages
	cr.ResultRaw = temp.ResultRaw

	if len(cr.ResultRaw) < 2 { // Check for empty or "{}"
		return nil
	}

	// Define a probe struct to detect the format of the 'result' field.
	type ResultProbe struct {
		Choices   *json.RawMessage `json:"choices"` // Check for presence of 'choices'
		ToolCalls *[]struct {
			ID string `json:"id"` // Check for presence of 'id' in tool_calls
		} `json:"tool_calls"`
	}

	var probe ResultProbe
	// We only care about whether this unmarshaling works and what fields are populated,
	// so we can ignore the error.
	_ = json.Unmarshal(cr.ResultRaw, &probe)

	// Case 1: Standard OpenAI format (has a "choices" array).
	if probe.Choices != nil {
		cr.IsLegacyResult = false
		return json.Unmarshal(cr.ResultRaw, &cr.ChatCompletionResponse)
	}

	// Case 2: Hybrid format (no "choices", but has modern tool calls with an "id").
	if probe.ToolCalls != nil && len(*probe.ToolCalls) > 0 && (*probe.ToolCalls)[0].ID != "" {
		cr.IsLegacyResult = false
		// Manually construct the ChatCompletionResponse since 'choices' is missing.
		var result struct {
			ToolCalls []ToolCall `json:"tool_calls"`
			Usage     Usage      `json:"usage"`
		}
		if err := json.Unmarshal(cr.ResultRaw, &result); err != nil {
			return fmt.Errorf("failed to parse hybrid response result: %w", err)
		}
		cr.ChatCompletionResponse.Choices = []Choice{
			{
				Message: ResponseMessage{
					Role:      "assistant",
					ToolCalls: result.ToolCalls,
				},
			},
		}
		cr.ChatCompletionResponse.Usage = result.Usage
		return nil
	}

	// Case 3: Fallback to legacy format.
	cr.IsLegacyResult = true
	return json.Unmarshal(cr.ResultRaw, &cr.LegacyResponse)
}

// ChatCompletionRequest is the complete payload sent to the Chat Completions API.
type ChatCompletionRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"` // Can contain ChatMessage or ToolMessage.
	Tools    []Tool    `json:"tools,omitempty"`
	Stream   bool      `json:"stream,omitempty"`
}

// UnmarshalJSON provides custom unmarshaling logic for the ChatCompletionRequest.
// This is necessary because the 'Messages' field is a slice of an interface type (Message),
// and the standard JSON library cannot determine which concrete struct to use for each element.
func (r *ChatCompletionRequest) UnmarshalJSON(data []byte) error {
	// Use an alias to avoid an infinite loop of recursive calls to this method.
	type Alias ChatCompletionRequest

	// Create a temporary struct where Messages is a slice of raw JSON.
	// This allows us to capture the top-level fields and the message objects
	// without unmarshaling the messages yet.
	temp := &struct {
		Messages []json.RawMessage `json:"messages"`
		*Alias
	}{
		Alias: (*Alias)(r),
	}

	if err := json.Unmarshal(data, &temp); err != nil {
		return fmt.Errorf("failed to unmarshal request shell: %w", err)
	}

	// Now, iterate through the raw message objects and unmarshal each one
	// into its correct concrete type.
	r.Messages = make([]Message, len(temp.Messages))
	for i, rawMsg := range temp.Messages {
		// First, probe the message to find its role.
		var probe struct {
			Role string `json:"role"`
		}
		if err := json.Unmarshal(rawMsg, &probe); err != nil {
			return fmt.Errorf("failed to probe message role: %w", err)
		}

		// Use the role to decide which struct to use.
		switch probe.Role {
		case "user", "system":
			var msg ChatMessage
			if err := json.Unmarshal(rawMsg, &msg); err != nil {
				return fmt.Errorf("failed to unmarshal ChatMessage: %w", err)
			}
			r.Messages[i] = msg
		case "assistant":
			// An assistant message could be a simple text response or a tool call request.
			// We need to probe for the presence of 'tool_calls' to differentiate.
			var toolCallProbe struct {
				ToolCalls []ToolCall `json:"tool_calls"`
			}
			// We can ignore the error here; we only care if the field exists.
			_ = json.Unmarshal(rawMsg, &toolCallProbe)

			if len(toolCallProbe.ToolCalls) > 0 {
				var msg ResponseMessage
				if err := json.Unmarshal(rawMsg, &msg); err != nil {
					return fmt.Errorf("failed to unmarshal ResponseMessage: %w", err)
				}
				r.Messages[i] = msg
			} else {
				var msg ChatMessage
				if err := json.Unmarshal(rawMsg, &msg); err != nil {
					return fmt.Errorf("failed to unmarshal assistant ChatMessage: %w", err)
				}
				r.Messages[i] = msg
			}
		case "tool":
			var msg ToolMessage
			if err := json.Unmarshal(rawMsg, &msg); err != nil {
				return fmt.Errorf("failed to unmarshal ToolMessage: %w", err)
			}
			r.Messages[i] = msg
		default:
			return fmt.Errorf("unknown message role found: %s", probe.Role)
		}
	}

	return nil
}

// ChatCompletionResponse is the top-level object returned by the API for the
// standard OpenAI-compatible format.
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

// Choice represents one of the possible completions generated by the model.
type Choice struct {
	Index        int             `json:"index"`
	Message      ResponseMessage `json:"message"`
	FinishReason string          `json:"finish_reason"` // e.g., "stop", "tool_calls".
}

// ResponseMessage is the message object returned by the model inside a Choice.
type ResponseMessage struct {
	Role             string     `json:"role"`    // Always "assistant".
	Content          *string    `json:"content"` // Can be null if calling tools.
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
}

// Implements the marker function that identifies it as a chat message
func (ResponseMessage) isMessage() {}

// Usage contains token count information for the request.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ListModels is unpacked into this type
type ModelsResponse map[string]*ModelInfo

// Model attributes struct
type ModelInfo struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Task        struct {
		Name        string `json:"name"`
		Description string `json:"description"`
	} `json:"task"`
	Tags       []string `json:"tags"`
	Properties struct {
		MaxBatchSize   int `json:"max_batch_size"`
		MaxTotalTokens int `json:"max_total_tokens"`
	} `json:"properties"`
	Source struct {
		URL string `json:"url"`
	} `json:"source"`
	Beta       bool                  `json:"beta"`
	Parameters map[string]*Parameter `json:"parameters"`
}

// =================================================================================
// API Response Structs
// These structs are designed to handle multiple response formats from the API.
// =================================================================================

// LegacyResponse matches the older, non-standard response format.
type LegacyResponse struct {
	Response  string           `json:"response"`
	ToolCalls []LegacyToolCall `json:"tool_calls"`
	Usage     Usage            `json:"usage"`
}

// LegacyToolCall defines the unique structure of a tool call in the legacy API format.
// which has a different structure from the standard OpenAI format.
type LegacyToolCall struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"` // Use RawMessage to hold the object
}
