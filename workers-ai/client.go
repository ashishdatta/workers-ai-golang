package workersai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

const (
	DefaultBaseURL = "https://api.cloudflare.com/client/v4"
)

type Client struct {
	BaseURL   string
	AccountID string
	APIToken  string
	HTTPClient *http.Client
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRequest struct {
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
}

type ChatResponse struct {
	Success bool     `json:"success"`
	Errors  []string `json:"errors"`
	Result  struct {
		Response string `json:"response"`
	} `json:"result"`
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
	}
}

func (c *Client) Chat(modelID string, messages []Message) (*ChatResponse, error) {
	url := fmt.Sprintf("%s/accounts/%s/ai/run/%s", c.BaseURL, c.AccountID, modelID)
	
	request := ChatRequest{
		Messages: messages,
		Stream:   false,
	}
	
	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	
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
	
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API returned status %d: %s", resp.StatusCode, string(body))
	}
	
	var response ChatResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}
	
	return &response, nil
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
