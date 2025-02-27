import { HumanMessage } from '@langchain/core/messages';
import { DynamicTool } from '@langchain/core/tools';
import {
    ChatDeepSeek,
    ChatDeepSeekInput,
} from '@langchain/deepseek';
import { MemorySaver } from '@langchain/langgraph';
import {
    InMemoryStore,
    MemorySaver as MemorySaverV2,
} from '@langchain/langgraph-checkpoint';
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import {
    ChatOpenAI,
    ChatOpenAIFields,
} from '@langchain/openai';

require("dotenv").config();

// ---------- Initialize Models ----------

// 1. Orchestrator agent using OpenAI ChatGPT
const configMain: ChatOpenAIFields = {
    modelName: process.env.OPENAI_MODEL ?? "gpt-3.5-turbo-0613", // or "gpt-4" if available
    temperature: 0,
    openAIApiKey: process.env.OPENAI_API_KEY, // your OpenAI API key
    configuration: {
    },
};
console.log('OPEN_AI', configMain)
const openAIModel = new ChatOpenAI(configMain);


// 2. DeepSeek model
const configDeepSeek: Partial<ChatDeepSeekInput> = {
    apiKey: process.env.DEEPSEEK_API_KEY,
    model: process.env.DEEPSEEK_MODEL ?? 'deepseek-chat',
};
console.log(`DEEPSEEK`, configDeepSeek)
const deepSeekModel = new ChatDeepSeek(configDeepSeek)


// 3. Local LM Studio model via its OpenAI-compatible API endpoint
// Make sure LM Studio is running as a server (e.g., http://localhost:8000/v1)
const localConfig: ChatOpenAIFields = {
    modelName: process.env.LM_STUDIO_MODEL, // e.g., "Llama-2-7b-chat"
    ...(process.env.LM_STUDIO_API_KEY && { openAIApiKey: process.env.LM_STUDIO_API_KEY }),           // LM Studio may require a key set in its config
    temperature: 0.7,
    configuration: {
        baseURL: process.env.LM_STUDIO_URL ?? "http://localhost:8000/v1", // LM Studio's API endpoint; adjust if needed
    }
};
console.log(`LOCAL LM STUDIO`, localConfig)
const localModel = new ChatOpenAI(localConfig);

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
const memory: any = new MemorySaver();

const memoryV3 = new InMemoryStore();
// check otehr packa
const memoryV2 = new MemorySaverV2();


// const memoryStore = new InMemoryStore();
// const memoryV2 = new MemorySaverV2({ store: memoryStore });
const writeConfig = {
    configurable: {
        thread_id: "1",
        checkpoint_ns: ""
    }
};
const checkpoint = {
    v: 1,
    ts: "2024-07-31T20:14:19.804150+00:00",
    id: "1ef4f797-8335-6428-8001-8a1503f9b875",
    channel_values: {
        my_key: "meow",
        node: "node"
    },
    channel_versions: {
        __start__: 2,
        my_key: 3,
        "start:node": 3,
        node: 3
    },
    versions_seen: {
        __input__: {},
        __start__: {
            __start__: 1
        },
        node: {
            "start:node": 2
        }
    },
    pending_sends: [],
}

const readConfig = {
    configurable: {
        thread_id: "1"
    }
};



// ---------- Running the Multi-Agent Conversation ----------
console.log(`Running graph`);

(async () => {


    await memoryV2.put(writeConfig, checkpoint, <any>{

    });
    console.log(await memoryV2.get(readConfig));


    // The orchestrator uses the OpenAI model and has access to the two tools.
    // It decides when to call ask_deepseek or ask_local_model based on the user query.
    const orchestratorAgent = createReactAgent({
        name: `graph`,
        llm: openAIModel,
        tools: [], // [askDeepSeekTool],// [askDeepSeekTool, askLocalTool],
        checkpointSaver: memoryV2, // ;________;
        // checkpointSaver: memoryV2,
        // Optionally, add a system prompt here to instruct the orchestrator's behavior.
    });

    // Example user question that might need both technical and creative inputs:
    const userQuestion = "Can you explain a complex math concept in simple terms with an analogy?";

    // The orchestrator agent processes the user input.
    const resultState = await orchestratorAgent.invoke({
        messages: [
            // new HumanMessage(userQuestion)
            {
                role: "user",
                content: "what is the weather in sf"
            }
        ],
    }, {
        recursionLimit: 2,
        configurable: {
            threadId: 1

        }
    });

    // Extract the final answer from the conversation state.
    const finalAnswer = resultState.messages[resultState.messages.length - 1].content;
    console.log("Final Answer:\n", finalAnswer);
})();


