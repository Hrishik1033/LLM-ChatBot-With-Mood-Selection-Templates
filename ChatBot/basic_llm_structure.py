from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage

load_dotenv()

# Use a standard text-generation model repository (not a GGUF repo)
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    temperature=0.2,
    task="text-generation" # Explicitly setting the task helps prevent routing issues
)

# This wraps it up as a clean chat model interface
model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content = 'You are a funny AI agent')
]

prompt = ''
while(prompt.lower() != 'exit'):
    prompt = input('You: ')

    messages.append(HumanMessage(content = prompt))

    if prompt.lower() == 'exit':
        break
    response = model.invoke(messages)

    messages.append(AIMessage(response.content))#This is done to preserve the message history, this is well orchestrated by langchain framework
    
    print('Assistant: ', response.content)