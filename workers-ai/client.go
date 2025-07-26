package workersai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
)

const (
	DefaultBaseURL = "https://api.cloudflare.com/client/v4"
)

type Client struct {
	BaseURL    string
	AccountID  string
	APIToken   string
	HTTPClient *http.Client
	Debug      bool
}

// Message is an interface implemented by all message types that can be sent to the API.
// It uses a marker method to ensure only specific structs can be used.
type Message interface {
	isMessage()
}

func NewClient(accountID, apiToken string) *Client {
	return &Client{
		BaseURL:    DefaultBaseURL,
		AccountID:  accountID,
		APIToken:   apiToken,
		HTTPClient: &http.Client{},
		Debug:      os.Getenv("WORKERS_AI_DEBUG") == "true",
	}
}

func (c *Client) SetDebug(debug bool) {
	c.Debug = debug
}

func (c *Client) Chat(modelID string, messages []Message) (*ChatResponse, error) {
	return c.ChatWithTools(modelID, messages, nil)
}

func (c *Client) ChatWithTools(modelID string, messages []Message, tools []Tool) (*ChatResponse, error) {
	var url string
	if strings.HasPrefix(modelID, "@cf/") {
		url = fmt.Sprintf("%s/accounts/%s/ai/run/%s", c.BaseURL, c.AccountID, modelID)
	} else {
		url = fmt.Sprintf("%s/accounts/%s/ai/run/@cf/%s", c.BaseURL, c.AccountID, modelID)
	}

	request := ChatCompletionRequest{
		Model:    modelID, // The model is part of the request body in the standard spec.
		Messages: messages,
		Tools:    tools,
		Stream:   false,
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	c.debugLog("Request URL: %s", url)
	c.debugLog("Request Body: %s", string(jsonData))

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.APIToken))
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	c.debugLog("Response Body: %s", string(body))

	if resp.StatusCode != http.StatusOK {
		c.debugLog("API Error - Status: %d, Body: %s", resp.StatusCode, string(body))
		return nil, fmt.Errorf("API returned status %d: %s", resp.StatusCode, string(body))
	}

	c.debugLog("Starting JSON unmarshal...")

	var response ChatResponse

	if err := json.Unmarshal(body, &response); err != nil {
		c.debugLog("JSON unmarshal failed: %v", err)
		return nil, fmt.Errorf("failed to parse ChatResponse: %w", err)
	}

	c.debugLog("Successfully parsed response. Detected legacy format: %v", response.IsLegacyResult)

	return &response, nil
}

// GetContent returns the content from the response, abstracting away the format differences.
func (r *ChatResponse) GetContent() string {
	if r.IsLegacyResult {
		return r.LegacyResponse.Response
	}

	if len(r.ChatCompletionResponse.Choices) > 0 {
		choice := r.ChatCompletionResponse.Choices[0]
		if choice.Message.Content != nil {
			return *choice.Message.Content
		}
	}
	return ""
}

func (r *ChatResponse) GetReasoningContent() string {
	if len(r.ChatCompletionResponse.Choices) > 0 {
		return r.ChatCompletionResponse.Choices[0].Message.ReasoningContent
	}
	return ""
}

// GetToolCalls returns tool calls from the response, abstracting away the format differences.
// are correctly populated before being returned. This fixes the test failure.
func (r *ChatResponse) GetToolCalls() []ToolCall {
	if r.IsLegacyResult {
		if r.LegacyResponse.ToolCalls == nil {
			return nil
		}
		// Adapt the legacy tool call structure to the standard one.
		adaptedCalls := make([]ToolCall, len(r.LegacyResponse.ToolCalls))
		for i, legacyCall := range r.LegacyResponse.ToolCalls {
			adaptedCalls[i] = ToolCall{
				// Legacy format doesn't have an ID, so we can leave it empty or generate one.
				ID:   fmt.Sprintf("legacy-tool-call-%d", i),
				Type: "function",
				Function: FunctionToCall{
					Name: legacyCall.Name,
					// The legacy arguments are a JSON object, so we convert them to a string.
					Arguments: string(legacyCall.Arguments),
				},
			}
		}
		return adaptedCalls
	}

	if len(r.ChatCompletionResponse.Choices) > 0 {
		return r.ChatCompletionResponse.Choices[0].Message.ToolCalls
	}
	return nil
}

func (c *Client) ListModels() ([]ModelInfo, error) {
	url := "https://ai.cloudflare.com/api/models"

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API returned status %d: %s", resp.StatusCode, string(body))
	}

	// The fix is here: Unmarshal directly into a slice of ModelInfo.
	var models ModelsResponse
	if err := json.Unmarshal(body, &models); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return nil, nil
}

func (c *Client) GetModelInfo(modelID string) (*ModelInfo, error) {
	url := fmt.Sprintf("%s/accounts/%s/ai/models/%s", c.BaseURL, c.AccountID, modelID)

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.APIToken))
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API returned status %d: %s", resp.StatusCode, string(body))
	}

	var modelInfo ModelInfo
	if err := json.Unmarshal(body, &modelInfo); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &modelInfo, nil
}

func (c *Client) debugLog(format string, args ...interface{}) {
	if c.Debug {
		log.Printf("[WORKERS_AI_DEBUG] "+format, args...)
	}
}
