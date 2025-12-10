from langchain_ollama import ChatOllama

print("Creating client...")
llm = ChatOllama(model="gemma2:2b")

print("Invoking...")
res = llm.invoke("Hello, how are you?")

print("DONE:", res)
