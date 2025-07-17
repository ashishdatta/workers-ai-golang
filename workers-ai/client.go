package workersai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
)

const (
	DefaultBaseURL = "https://api.cloudflare.com/client/v4"
)

type Client struct {
	BaseURL   string
	AccountID string
	APIToken  string
	HTTPClient *http.Client
	Debug     bool
}

type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type Message struct {
	Role      string     `json:"role"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string    `json:"tool_call_id,omitempty"`
}

type ChatRequest struct {
	Messages []Message      `json:"messages"`
	Tools    []FunctionTool `json:"tools,omitempty"`
	Stream   bool           `json:"stream"`
}

type Choice struct {
	Index   int `json:"index"`
	Message struct {
		Role             string     `json:"role"`
		ReasoningContent string     `json:"reasoning_content,omitempty"`
		Content          *string    `json:"content"`
		ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	} `json:"message"`
	LogProbs     interface{} `json:"logprobs"`
	FinishReason string      `json:"finish_reason"`
	StopReason   string      `json:"stop_reason,omitempty"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type OpenAIResponse struct {
	ID           string      `json:"id"`
	Object       string      `json:"object"`
	Created      int64       `json:"created"`
	Model        string      `json:"model"`
	Choices      []Choice    `json:"choices"`
	Usage        Usage       `json:"usage"`
	PromptLogProbs interface{} `json:"prompt_logprobs"`
}

type ChatResponse struct {
	Success  bool              `json:"success"`
	Errors   []string          `json:"errors"`
	Messages []interface{}     `json:"messages"`
	Result   OpenAIResponse    `json:"result"`
	
	// Legacy fields for backward compatibility - will be populated from Result
	LegacyResult struct {
		Response  string     `json:"response,omitempty"`
		ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	} `json:"-"`
}

type Parameter struct {
	Type        string `json:"type"`
	Description string `json:"description"`
	Default     interface{} `json:"default,omitempty"`
	Minimum     interface{} `json:"minimum,omitempty"`
	Maximum     interface{} `json:"maximum,omitempty"`
	Enum        []string `json:"enum,omitempty"`
	Items       *Parameter `json:"items,omitempty"`
}

type FunctionTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string `json:"name"`
		Description string `json:"description"`
		Parameters  struct {
			Type       string                `json:"type"`
			Required   []string              `json:"required"`
			Properties map[string]*Parameter `json:"properties"`
		} `json:"parameters"`
	} `json:"function"`
}

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
	Beta bool `json:"beta"`
	Parameters map[string]*Parameter `json:"parameters"`
}

type ModelsResponse map[string]*ModelInfo

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

func (c *Client) debugLog(format string, args ...interface{}) {
	if c.Debug {
		log.Printf("[WORKERS_AI_DEBUG] "+format, args...)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

func (c *Client) Chat(modelID string, messages []Message) (*ChatResponse, error) {
	return c.ChatWithTools(modelID, messages, nil)
}

func (c *Client) ChatWithTools(modelID string, messages []Message, tools []FunctionTool) (*ChatResponse, error) {
	url := fmt.Sprintf("%s/accounts/%s/ai/run/%s", c.BaseURL, c.AccountID, modelID)
	
	request := ChatRequest{
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
	
	c.debugLog("Request Headers: Authorization=Bearer %s..., Content-Type=%s", 
		c.APIToken[:min(len(c.APIToken), 10)], req.Header.Get("Content-Type"))
	
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()
	
	c.debugLog("Response Status: %d %s", resp.StatusCode, resp.Status)
	c.debugLog("Response Headers: %v", resp.Header)
	
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
	
	// First, let's try to unmarshal into a generic map to see the actual structure
	var rawResponse map[string]interface{}
	if err := json.Unmarshal(body, &rawResponse); err != nil {
		c.debugLog("Raw JSON unmarshal failed: %v", err)
		return nil, fmt.Errorf("failed to parse raw response: %w", err)
	}
	
	c.debugLog("Raw response structure:")
	for key, value := range rawResponse {
		c.debugLog("  %s: %T = %v", key, value, truncateString(fmt.Sprintf("%v", value), 200))
	}
	
	var response ChatResponse
	if err := json.Unmarshal(body, &response); err != nil {
		c.debugLog("JSON unmarshal failed: %v", err)
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}
	
	c.debugLog("JSON unmarshal successful")
	c.debugLog("Parsed response - Success: %t, Errors: %v, Result.ID: %s, Result.Model: %s, Result.Choices: %d", 
		response.Success, response.Errors, response.Result.ID, response.Result.Model, len(response.Result.Choices))
	
	if len(response.Result.Choices) > 0 {
		choice := response.Result.Choices[0]
		contentStr := "<nil>"
		if choice.Message.Content != nil {
			contentStr = *choice.Message.Content
		}
		c.debugLog("First choice - Role: %s, Content: %s, ReasoningContent: %s, ToolCalls: %d", 
			choice.Message.Role, 
			truncateString(contentStr, 100),
			truncateString(choice.Message.ReasoningContent, 100),
			len(choice.Message.ToolCalls))
	}
	
	c.debugLog("Populating legacy fields...")
	// Populate legacy fields for backward compatibility
	response.populateLegacyFields()
	
	c.debugLog("Legacy fields populated - LegacyResult.Response: %s, LegacyResult.ToolCalls: %d", 
		truncateString(response.LegacyResult.Response, 100), len(response.LegacyResult.ToolCalls))
	
	return &response, nil
}

// GetContent returns the content from the first choice, with fallback to legacy Result.Response
func (r *ChatResponse) GetContent() string {
	if len(r.Result.Choices) > 0 {
		if r.Result.Choices[0].Message.Content != nil && *r.Result.Choices[0].Message.Content != "" {
			return *r.Result.Choices[0].Message.Content
		}
		if r.Result.Choices[0].Message.ReasoningContent != "" {
			return r.Result.Choices[0].Message.ReasoningContent
		}
	}
	return r.LegacyResult.Response
}

// GetToolCalls returns tool calls from the first choice, with fallback to legacy Result.ToolCalls
func (r *ChatResponse) GetToolCalls() []ToolCall {
	if len(r.Result.Choices) > 0 {
		return r.Result.Choices[0].Message.ToolCalls
	}
	return r.LegacyResult.ToolCalls
}

// GetReasoningContent returns the reasoning content from the first choice
func (r *ChatResponse) GetReasoningContent() string {
	if len(r.Result.Choices) > 0 {
		return r.Result.Choices[0].Message.ReasoningContent
	}
	return ""
}

// populateLegacyFields populates the legacy Result fields for backward compatibility
func (r *ChatResponse) populateLegacyFields() {
	if len(r.Result.Choices) > 0 {
		choice := r.Result.Choices[0]
		
		// Set response content (prefer actual content over reasoning content)
		if choice.Message.Content != nil && *choice.Message.Content != "" {
			r.LegacyResult.Response = *choice.Message.Content
		} else if choice.Message.ReasoningContent != "" {
			r.LegacyResult.Response = choice.Message.ReasoningContent
		}
		
		// Set tool calls
		r.LegacyResult.ToolCalls = choice.Message.ToolCalls
	}
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
	
	var response ModelsResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}
	
	// Convert map to slice
	models := make([]ModelInfo, 0, len(response))
	for name, modelInfo := range response {
		if modelInfo != nil {
			modelInfo.Name = name // Set the name from the map key
			models = append(models, *modelInfo)
		}
	}
	
	return models, nil
}
