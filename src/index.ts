import { HumanMessage } from '@langchain/core/messages';
import { DynamicTool } from '@langchain/core/tools';
import { ChatDeepSeek } from '@langchain/deepseek';
import { MemorySaver } from '@langchain/langgraph';
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { ChatOpenAI } from '@langchain/openai';

// ---------- Initialize Models ----------

// 1. Orchestrator agent using OpenAI ChatGPT
const openAIModel = new ChatOpenAI({
    modelName: process.env.OPENAI_MODEL ?? "gpt-3.5-turbo-0613", // or "gpt-4" if available
    temperature: 0,
    openAIApiKey: process.env.OPENAI_API_KEY, // your OpenAI API key
});

// 2. DeepSeek model
const deepSeekModel = new ChatDeepSeek({
    apiKey: process.env.DEEPSEEK_API_KEY,
    model: process.env.DEEPSEEK_MODEL ?? 'deepseek-chat',
})

// 3. Local LM Studio model via its OpenAI-compatible API endpoint
// Make sure LM Studio is running as a server (e.g., http://localhost:8000/v1)
const localModel = new ChatOpenAI(
    {
        modelName: process.env.LM_STUDIO_MODEL, // e.g., "Llama-2-7b-chat"
        ...(process.env.LM_STUDIO_API_KEY && { openAIApiKey: process.env.LM_STUDIO_API_KEY }),           // LM Studio may require a key set in its config
        temperature: 0.7,
        configuration: {
            baseURL: process.env.LM_STUDIO_URL ?? "http://localhost:8000/v1", // LM Studio's API endpoint; adjust if needed
        }
    },
);

// ---------- Wrap Sub-Agent Calls as Tools ----------


// Tool to call DeepSeek (via Ollama)
const askDeepSeekTool = new DynamicTool({
    name: "ask_deepseek",
    description: "Use the DeepSeek model (via Ollama) for technical reasoning.",
    func: async (input: string) => {
        const response = await deepSeekModel.invoke(input);
        return typeof response === "string" ? response : response.toString();
    },
});

// Tool to call the local LM Studio model
const askLocalTool = new DynamicTool({
    name: "ask_local_model",
    description: "Use the local LM Studio model for creative explanations.",
    func: async (input: string) => {
        const response = await localModel.invoke([new HumanMessage(input)]);
        return response.content;
    },
});


// ---------- Create the Orchestrator Agent ----------

// MemorySaver keeps a running context of the conversation.
const memory = new MemorySaver();

// The orchestrator uses the OpenAI model and has access to the two tools.
// It decides when to call ask_deepseek or ask_local_model based on the user query.
const orchestratorAgent = createReactAgent({
    llm: openAIModel,
    tools: [askDeepSeekTool, askLocalTool],
    checkpointSaver: memory,
    // Optionally, add a system prompt here to instruct the orchestrator's behavior.
});

// ---------- Running the Multi-Agent Conversation ----------

(async () => {
    // Example user question that might need both technical and creative inputs:
    const userQuestion = "Can you explain a complex math concept in simple terms with an analogy?";

    // The orchestrator agent processes the user input.
    const resultState = await orchestratorAgent.invoke({
        messages: [new HumanMessage(userQuestion)],
    });

    // Extract the final answer from the conversation state.
    const finalAnswer = resultState.messages[resultState.messages.length - 1].content;
    console.log("Final Answer:\n", finalAnswer);
})();
