import {
    BaseMessage,
    HumanMessage,
} from '@langchain/core/messages';
import { DynamicTool } from '@langchain/core/tools';
import { ChatDeepSeek } from '@langchain/deepseek';
import {
    Annotation,
    MemorySaver,
    StateGraph,
} from '@langchain/langgraph';
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import {
    ChatOpenAI,
    ChatOpenAIFields,
} from '@langchain/openai';

require("dotenv").config();

// ---------- Initialize Models ----------

// OpenAI ChatGPT
const openAIModel = new ChatOpenAI({
    modelName: process.env.OPENAI_MODEL ?? "gpt-3.5-turbo-0613",
    temperature: 0,
    openAIApiKey: process.env.OPENAI_API_KEY,
});

// DeepSeek Model
const deepSeekModel = new ChatDeepSeek({
    apiKey: process.env.DEEPSEEK_API_KEY,
    model: process.env.DEEPSEEK_MODEL ?? "deepseek-chat",
});

// Local Model
const localConfig: ChatOpenAIFields = {
    modelName: process.env.LM_STUDIO_MODEL,
    ...(process.env.LM_STUDIO_API_KEY && { openAIApiKey: process.env.LM_STUDIO_API_KEY }),           // LM Studio may require a key set in its config
    temperature: 0.7,
    configuration: {
        baseURL: process.env.LM_STUDIO_URL ?? "http://localhost:8000/v1", // LM Studio's API endpoint; adjust if needed
    }
};
const localModel = new ChatOpenAI(localConfig);


// ---------- Define Tools ----------
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



// ---------- Define Memory ----------
const memory = new MemorySaver();

// ---------- Define and Compile Graph ----------
export function makeGraph() {
    console.log("🚀 Creating orchestrator agent...");

    // Define the orchestrator agent
    const orchestratorAgent = createReactAgent({
        llm: openAIModel,
        tools: [askDeepSeekTool, askLocalTool],
        checkpointSaver: memory,
    });

    const StateAnnotation = Annotation.Root({
        sentiment: Annotation<string>,
        messages: Annotation<BaseMessage[]>({
            reducer: (left: BaseMessage[], right: BaseMessage | BaseMessage[]) => {

                if (Array.isArray(right)) {
                    return left.concat(right);
                }
                return left.concat([right]);
            },
            default: () => [],
        }),
    });

    console.log("🔄 Creating state graph...");
    const graph = new StateGraph(StateAnnotation);

    graph
        .addNode("orchestrate", orchestratorAgent)
        .addEdge("__start__", "orchestrate");

    console.log("✅ Compiling state graph...");
    const compiledGraph = graph.compile();

    console.log("✅ Graph successfully created!");
    return compiledGraph;

}

