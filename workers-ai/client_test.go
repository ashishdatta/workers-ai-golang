package workersai

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
)

func TestClient_Chat(t *testing.T) {
	mockResponse := ChatResponse{
		Success: true,
		Result: struct {
			Response string `json:"response"`
		}{
			Response: "Hello! How can I help you today?",
		},
	}
	
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("Expected POST request, got %s", r.Method)
		}
		
		if r.Header.Get("Authorization") != "Bearer test-token" {
			t.Errorf("Expected Authorization header with Bearer token")
		}
		
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("Expected Content-Type to be application/json")
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(mockResponse)
	}))
	defer server.Close()
	
	client := NewClient("test-account", "test-token")
	client.BaseURL = server.URL
	
	messages := []Message{
		{Role: "system", Content: "You are a helpful assistant"},
		{Role: "user", Content: "Hello"},
	}
	
	response, err := client.Chat(ModelLlama4Scout17B, messages)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	
	if response.Result.Response != "Hello! How can I help you today?" {
		t.Errorf("Expected response 'Hello! How can I help you today?', got %s", response.Result.Response)
	}
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

func TestClient_ListModels(t *testing.T) {
	mockResponse := ModelsResponse{
		"@cf/meta/llama-3-8b-instruct": &ModelInfo{
			Description: "First test model",
			Task: struct {
				Name        string `json:"name"`
				Description string `json:"description"`
			}{
				Name:        "text-generation",
				Description: "Generates text",
			},
			Parameters: map[string]*Parameter{
				"max_tokens": {
					Type:        "integer",
					Description: "Maximum tokens",
					Default:     256,
				},
			},
		},
		"@cf/meta/llama-3-70b-instruct": &ModelInfo{
			Description: "Second test model",
			Task: struct {
				Name        string `json:"name"`
				Description string `json:"description"`
			}{
				Name:        "text-generation", 
				Description: "Generates text",
			},
			Parameters: map[string]*Parameter{
				"temperature": {
					Type:        "number",
					Description: "Controls randomness",
					Default:     0.15,
				},
			},
		},
	}
	
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" {
			t.Errorf("Expected GET request, got %s", r.Method)
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(mockResponse)
	}))
	defer server.Close()
	
	client := NewClient("test-account", "test-token")
	client.BaseURL = server.URL
	
	models, err := client.ListModels()
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	
	if len(models) != 2 {
		t.Errorf("Expected 2 models, got %d", len(models))
	}
	
	// Check that model names are set from map keys
	modelNames := make(map[string]bool)
	for _, model := range models {
		modelNames[model.Name] = true
	}
	
	if !modelNames["@cf/meta/llama-3-8b-instruct"] {
		t.Error("Expected model '@cf/meta/llama-3-8b-instruct' not found")
	}
	
	if !modelNames["@cf/meta/llama-3-70b-instruct"] {
		t.Error("Expected model '@cf/meta/llama-3-70b-instruct' not found")
	}
}

func TestClient_Chat_Integration(t *testing.T) {
	accountID := os.Getenv("CLOUDFLARE_ACCOUNT_ID")
	apiToken := os.Getenv("CLOUDFLARE_AUTH_TOKEN")
	
	if accountID == "" || apiToken == "" {
		t.Skip("Skipping integration test: CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_AUTH_TOKEN not set")
	}
	
	client := NewClient(accountID, apiToken)
	
	messages := []Message{
		{Role: "system", Content: "You are a helpful assistant. Keep responses brief."},
		{Role: "user", Content: "Say 'Hello World' and nothing else."},
	}
	
	response, err := client.Chat(ModelLlama4Scout17B, messages)
	if err != nil {
		t.Fatalf("Integration test failed: %v", err)
	}
	
	if response.Result.Response == "" {
		t.Error("Expected non-empty response")
	}
	
	t.Logf("AI Response: %s", response.Result.Response)
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