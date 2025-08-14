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
	apiToken := os.Getenv("CLOUDFLARE_API_TOKEN")

	if accountID == "" || apiToken == "" {
		log.Fatal("Please set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN environment variables")
	}

	client := workersai.NewClient(accountID, apiToken)

	// Enable debug logging - can also be enabled with WORKERS_AI_DEBUG=true environment variable
	client.SetDebug(false)

	fmt.Println("\n--- Simple Chat Example ---")
	chatMessages := []workersai.Message{
		workersai.ChatMessage{Role: "system", Content: "You are a friendly assistant"},
		workersai.ChatMessage{Role: "user", Content: "Why is pizza so good?"},
	}

	p := workersai.ModelParameters{
		Temperature: 0.1,
	}
	// Using a known model from the library constants
	chatResponse, err := client.Chat(workersai.ModelQwen330ba3b, chatMessages, &p)
	if err != nil {
		log.Printf("Error sending chat request: %v", err)
	} else {
		fmt.Printf("AI Response: %s\n", chatResponse.GetContent())
		if reasoning := chatResponse.GetReasoningContent(); reasoning != "" {
			fmt.Printf("Reasoning: %s\n", reasoning)
		}

		// Correctly print usage based on the response format
		if chatResponse.IsLegacyResult {
			fmt.Printf("Usage: %d prompt + %d completion = %d total tokens\n",
				chatResponse.LegacyResponse.Usage.PromptTokens,
				chatResponse.LegacyResponse.Usage.CompletionTokens,
				chatResponse.LegacyResponse.Usage.TotalTokens)
		} else {
			fmt.Printf("Usage: %d prompt + %d completion = %d total tokens\n",
				chatResponse.ChatCompletionResponse.Usage.PromptTokens,
				chatResponse.ChatCompletionResponse.Usage.CompletionTokens,
				chatResponse.ChatCompletionResponse.Usage.TotalTokens)
		}
	}

	fmt.Println("\n--- Tool Calling Example ---")

	// Define a tool using the new, type-safe Tool struct
	tools := []workersai.Tool{
		{
			Type: "function",
			Function: workersai.FunctionDefinition{
				Name:        "get_weather",
				Description: "Get the current weather in a given location",
				Parameters: workersai.FunctionParameters{
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
		workersai.ChatMessage{Role: "system", Content: "You are a helpful assistant with access to weather information"},
		workersai.ChatMessage{Role: "user", Content: "What's the weather like in San Francisco?"},
	}

	toolResponse, err := client.ChatWithTools(workersai.ModelQwen330ba3b, toolMessages, tools, nil)
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

		// Correctly print usage for the tool response
		if toolResponse.IsLegacyResult {
			fmt.Printf("Usage: %d prompt + %d completion = %d total tokens\n",
				toolResponse.LegacyResponse.Usage.PromptTokens,
				toolResponse.LegacyResponse.Usage.CompletionTokens,
				toolResponse.LegacyResponse.Usage.TotalTokens)
		} else {
			fmt.Printf("Usage: %d prompt + %d completion = %d total tokens\n",
				toolResponse.ChatCompletionResponse.Usage.PromptTokens,
				toolResponse.ChatCompletionResponse.Usage.CompletionTokens,
				toolResponse.ChatCompletionResponse.Usage.TotalTokens)
		}
	}

	// --- List Models Example ---
	// fmt.Println("\n--- List Available Models ---")
	// models, err := client.ListModels()
	// if err != nil {
	// 	log.Printf("Error listing models: %v", err)
	// } else {
	// 	fmt.Printf("Found %d models:\n", len(models))
	// 	for i, model := range models {
	// 		if i >= 5 { // Show only the first 5 models
	// 			fmt.Printf("... and %d more\n", len(models)-5)
	// 			break
	// 		}
	// 		fmt.Printf("- %s: %s\n", model.Name, model.Description)
	// 	}
	// }
}
