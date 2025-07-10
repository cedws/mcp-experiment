package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/alecthomas/chroma/v2/quick"
	"github.com/charmbracelet/huh"
	"github.com/charmbracelet/lipgloss"
	"github.com/mark3labs/mcp-go/client"
	mcpclient "github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

const defaultModel = "google/gemini-2.5-flash"

var systemMessages = []openai.ChatCompletionMessageParamUnion{
	openai.SystemMessage("To be a fast and efficient agent, batch tool calls together."),
	openai.SystemMessage("Do everything using a Python sandbox. Don't use built-in tool calling, use the Python sandbox."),
	openai.SystemMessage("Don't try to calculate yourself or retrieve results from memory. You compute everything."),
	openai.SystemMessage("Output the result and ONLY the result."),
}

func print(s string, a ...any) {
	fmt.Printf(s+"\n", a...)
}

var (
	codeBoxStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("62")).
			Padding(1, 2).
			MarginLeft(2)

	resultBoxStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("42")).
			Padding(1, 2).
			MarginLeft(2)
)

func printCodeBox(content, language string) {
	var buf strings.Builder
	if err := quick.Highlight(&buf, content, language, "terminal256", "monokai"); err != nil {
		buf.WriteString(content)
	}

	styledBox := codeBoxStyle.
		BorderTop(true).
		BorderTopForeground(lipgloss.Color("62")).
		Render(buf.String())

	fmt.Println(styledBox)
}

func printResultBox(content string) {
	fmt.Println(resultBoxStyle.Render(content))
}

func main() {
	ctx := context.Background()

	mcpClient, err := client.NewStreamableHttpClient("http://127.0.0.1:5555/mcp")
	if err != nil {
		log.Fatalf("Failed to create MCP client: %v", err)
	}
	defer mcpClient.Close()

	if err := mcpClient.Start(ctx); err != nil {
		log.Fatalf("Failed to start MCP client: %v", err)
	}

	toolsResult := toolList(ctx, mcpClient)
	toolsSchema := convertToolsSchema(toolsResult)

	apiKey, ok := os.LookupEnv("OPENAI_API_KEY")
	if !ok {
		log.Fatal("OPENAI_API_KEY environment variable not set")
	}

	openaiClient := openai.NewClient(
		option.WithBaseURL("https://openrouter.ai/api/v1"),
		option.WithAPIKey(apiKey),
	)

	models, err := fetchModels(ctx, openaiClient)
	if err != nil {
		log.Fatalf("Failed to fetch models: %v", err)
	}

	question, model, err := showForm(ctx, models)
	if err != nil {
		log.Fatalf("Failed to show form: %v", err)
	}

	print("Query: %s", question)

	params := openai.ChatCompletionNewParams{
		Tools:    toolsSchema,
		Model:    model,
		Messages: append(systemMessages, openai.UserMessage(question)),
	}

	for {
		completion, err := openaiClient.Chat.Completions.New(ctx, params)
		if err != nil {
			log.Fatalf("Failed to create chat completion: %v", err)
		}

		if completion.Choices[0].Message.Content != "" {
			printResultBox(completion.Choices[0].Message.Content)
		}

		toolCalls := completion.Choices[0].Message.ToolCalls
		if len(toolCalls) == 0 {
			break
		}

		params.Messages = append(
			params.Messages,
			completion.Choices[0].Message.ToParam(),
		)

		for _, toolCall := range toolCalls {
			result, err := callTool(ctx, mcpClient, toolCall)
			if err != nil {
				log.Fatalf("Failed to call tool: %v", err)
			}

			params.Messages = append(
				params.Messages,
				openai.ToolMessage(result, toolCall.ID),
			)
		}
	}
}

func showForm(ctx context.Context, models []string) (string, string, error) {
	var (
		question string
		model    = defaultModel
	)

	form := huh.NewForm(
		huh.NewGroup(
			huh.NewInput().
				Title("Enter a task").
				Value(&question),
			huh.NewSelect[string]().
				Title("Select a model").
				Value(&model).
				Height(10).
				Options(huh.NewOptions(models...)...),
		),
	)

	if err := form.RunWithContext(ctx); err != nil {
		log.Fatalf("Failed to run input: %v", err)
	}

	return question, model, nil
}

func fetchModels(ctx context.Context, openaiClient openai.Client) (res []string, err error) {
	models := openaiClient.Models.ListAutoPaging(ctx)

	for {
		res = append(res, models.Current().ID)

		switch {
		case models.Next():
			continue
		case models.Err() != nil:
			return nil, models.Err()
		default:
			return res, nil
		}
	}
}

func convertToolsSchema(tools *mcp.ListToolsResult) []openai.ChatCompletionToolParam {
	var openaiTools []openai.ChatCompletionToolParam

	for _, tool := range tools.Tools {
		schema := map[string]any{
			"type": "object",
		}

		if len(tool.InputSchema.Properties) > 0 {
			schema["properties"] = tool.InputSchema.Properties
		} else {
			schema["properties"] = map[string]any{}
		}

		if len(tool.InputSchema.Required) > 0 {
			schema["required"] = tool.InputSchema.Required
		}

		openaiTool := openai.ChatCompletionToolParam{
			Function: openai.FunctionDefinitionParam{
				Name:        tool.Name,
				Description: openai.String(tool.Description),
				Parameters:  openai.FunctionParameters(schema),
			},
		}

		openaiTools = append(openaiTools, openaiTool)
	}

	return openaiTools
}

func toolList(ctx context.Context, mcpClient *mcpclient.Client) *mcp.ListToolsResult {
	initRequest := mcp.InitializeRequest{
		Request: mcp.Request{
			Method: "initialize",
		},
		Params: mcp.InitializeParams{
			ProtocolVersion: mcp.LATEST_PROTOCOL_VERSION,
			Capabilities: mcp.ClientCapabilities{
				Experimental: map[string]any{},
			},
			ClientInfo: mcp.Implementation{
				Name:    "mcp-client",
				Version: "1.0.0",
			},
		},
	}

	if _, err := mcpClient.Initialize(ctx, initRequest); err != nil {
		log.Fatalf("Failed to initialize MCP client: %v", err)
	}

	toolsResult, err := mcpClient.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		log.Fatalf("Failed to list tools: %v", err)
	}
	if len(toolsResult.Tools) == 0 {
		log.Fatal("No tools available from MCP server")
	}

	return toolsResult
}

func callTool(ctx context.Context, mcpClient *mcpclient.Client, toolCall openai.ChatCompletionMessageToolCall) (string, error) {
	var args map[string]any

	if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
		return "", fmt.Errorf("failed to unmarshal tool arguments: %v", err)
	}

	switch toolCall.Function.Name {
	case "sandbox_run_code":
		printCodeBox(args["code"].(string), "python")
	}

	mcpToolRequest := mcp.CallToolRequest{
		Request: mcp.Request{
			Method: "tools/call",
		},
		Params: mcp.CallToolParams{
			Name:      toolCall.Function.Name,
			Arguments: args,
		},
	}

	toolResult, err := mcpClient.CallTool(ctx, mcpToolRequest)
	if err != nil {
		return "", fmt.Errorf("failed to call tool: %v", err)
	}

	var resultText string

	if len(toolResult.Content) > 0 {
		if textContent, ok := mcp.AsTextContent(toolResult.Content[0]); ok {
			resultText = textContent.Text
		} else {
			resultText = fmt.Sprintf("%v", toolResult.Content[0])
		}
	}

	return resultText, nil
}
