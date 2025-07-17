package main

import (
	"fmt"
	"log"
	"os"

	workersai "github.com/ashishdatta/workers-ai-golang/workers-ai"
)

func main() {
	fmt.Println("Cloudflare Workers AI Go Client Example")
	
	accountID := os.Getenv("CLOUDFLARE_ACCOUNT_ID")
	apiToken := os.Getenv("CLOUDFLARE_AUTH_TOKEN")
	
	if accountID == "" || apiToken == "" {
		log.Fatal("Please set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_AUTH_TOKEN environment variables")
	}
	
	client := workersai.NewClient(accountID, apiToken)
	
	// Enable debug logging - can also be enabled with WORKERS_AI_DEBUG=true environment variable
	client.SetDebug(false)
	
	// Get model information
	/* modelInfo, err := client.GetModelInfo(workersai.ModelLlama4Scout17B)
	if err != nil {
		log.Printf("Error getting model info: %v", err)
	} else {
		fmt.Printf("Model: %s\n", modelInfo.Name)
		fmt.Printf("Description: %s\n", modelInfo.Description)
		fmt.Printf("Task: %s\n", modelInfo.Task.Name)
		fmt.Printf("Max Tokens: %d\n", modelInfo.Properties.MaxTotalTokens)
	} */
	
	// Chat with the model
	messages := []workersai.Message{
		{Role: "system", Content: "You are a friendly assistant"},
		{Role: "user", Content: "Why is pizza so good?"},
	}
	
	//response, err := client.Chat(workersai.ModelLlama4Scout17B, messages)
	response, err := client.Chat(workersai.ModelQwen330ba3b, messages)
	if err != nil {
		log.Printf("Error sending chat request: %v", err)
	} else {
		fmt.Printf("AI Response: %s\n", response.GetContent())
		if reasoning := response.GetReasoningContent(); reasoning != "" {
			fmt.Printf("Reasoning: %s\n", reasoning)
		}
		fmt.Printf("Usage: %d prompt + %d completion = %d total tokens\n", 
			response.Result.Usage.PromptTokens, response.Result.Usage.CompletionTokens, response.Result.Usage.TotalTokens)
	}
	
	// Example of tool calling
	fmt.Println("\n--- Tool Calling Example ---")
	
	// Define a simple function tool
	tools := []workersai.FunctionTool{
		{
			Type: "function",
			Function: struct {
				Name        string `json:"name"`
				Description string `json:"description"`
				Parameters  struct {
					Type       string                      `json:"type"`
					Required   []string                    `json:"required"`
					Properties map[string]*workersai.Parameter `json:"properties"`
				} `json:"parameters"`
			}{
				Name:        "get_weather",
				Description: "Get the current weather in a given location",
				Parameters: struct {
					Type       string                      `json:"type"`
					Required   []string                    `json:"required"`
					Properties map[string]*workersai.Parameter `json:"properties"`
				}{
					Type:     "object",
					Required: []string{"location"},
					Properties: map[string]*workersai.Parameter{
						"location": {
							Type:        "string",
							Description: "The city and state, e.g. San Francisco, CA",
						},
						"unit": {
							Type:        "string",
							Description: "The unit of temperature",
							Enum:        []string{"celsius", "fahrenheit"},
						},
					},
				},
			},
		},
	}
	
	toolMessages := []workersai.Message{
		{Role: "system", Content: "You are a helpful assistant with access to weather information"},
		{Role: "user", Content: "What's the weather like in San Francisco?"},
	}
	
  //toolResponse, err := client.ChatWithTools(workersai.ModelLlama4Scout17B, toolMessages, tools)
  toolResponse, err := client.ChatWithTools(workersai.ModelQwen330ba3b, toolMessages, tools)
	if err != nil {
		log.Printf("Error sending tool calling request: %v", err)
	} else {
		toolCalls := toolResponse.GetToolCalls()
		if len(toolCalls) > 0 {
			fmt.Printf("Tool calls requested: %d\n", len(toolCalls))
			for _, toolCall := range toolCalls {
				fmt.Printf("- Function: %s\n", toolCall.Function.Name)
				fmt.Printf("  Arguments: %s\n", toolCall.Function.Arguments)
			}
		} else {
			fmt.Printf("AI Response: %s\n", toolResponse.GetContent())
		}
		
		if reasoning := toolResponse.GetReasoningContent(); reasoning != "" {
			fmt.Printf("Reasoning: %s\n", reasoning)
		}
		
		fmt.Printf("Usage: %d prompt + %d completion = %d total tokens\n", 
			toolResponse.Result.Usage.PromptTokens, toolResponse.Result.Usage.CompletionTokens, toolResponse.Result.Usage.TotalTokens)
	}
	
	// List available models
	// models, err := client.ListModels()
	// if err != nil {
	// 	log.Printf("Error listing models: %v", err)
	// } else {
	// 	fmt.Printf("\nAvailable models (%d):\n", len(models))
	// 	for i, model := range models {
	// 		fmt.Printf("%d. %s - %s\n", i+1, model.Name, model.Description)
	// 		if i >= 4 { // Show only first 5 models
	// 			fmt.Printf("... and %d more\n", len(models)-5)
	// 			break
	// 		}
	// 	}
}
