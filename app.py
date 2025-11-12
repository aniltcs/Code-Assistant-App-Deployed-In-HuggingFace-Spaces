import gradio as gr
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------
# 🚀 1. Initialize Groq LLM
# ---------------------------------------------------------
llm_instances = {}  # store models to avoid reinitialization

def get_groq_model(api_key: str, model_name: str):
    key = f"{model_name}_{api_key}"
    if key not in llm_instances:
        llm_instances[key] = ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=0.2,
            max_tokens=300
        )
    return llm_instances[key]

# ---------------------------------------------------------
# 🧩 2. Prompt Template
# ---------------------------------------------------------
prompt_template = PromptTemplate.from_template(
    """You are an expert software developer and only respond to programming/code-related prompts.
    - If the prompt is unrelated to programming, respond: "⚠️ I only answer code-related queries."

    ### Instruction:
    {user_prompt}

    ### Your Response:
    """
)

# ---------------------------------------------------------
# ⚙️ 3. Function to generate code
# ---------------------------------------------------------
def generate_code(api_key, model_name, user_prompt):
    if not api_key.strip():
        return "⚠️ Please provide your Groq API key."
    
    llm = get_groq_model(api_key, model_name)
    chain = prompt_template | llm
    response = chain.invoke({"user_prompt": user_prompt})
    return response.content

# ---------------------------------------------------------
# 🎨 4. Gradio UI
# ---------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 💻 Code Assistant via Groq + LangChain")
    gr.Markdown("This app uses a Groq-supported LLM for code generation and explanation.")

    with gr.Row():
        api_key_input = gr.Textbox(label="🔑 Groq API Key", type="password", placeholder="Enter your Groq API key")
        model_name_input = gr.Textbox(label="Model Name", value="llama-3.3-70b-versatile")

    prompt_input = gr.Textbox(label="🧠 Enter your prompt:", placeholder="Write a Python function to reverse a string", lines=5)
    generate_button = gr.Button("🚀 Generate Code")
    output_box = gr.Markdown(label="🧩 Generated Code / Explanation")
    
    generate_button.click(
        fn=generate_code,
        inputs=[api_key_input, model_name_input, prompt_input],
        outputs=[output_box]
    )

# ---------------------------------------------------------
# 5. Launch app
# ---------------------------------------------------------
demo.launch(share=True)


