# Understanding LCEL from the ground up

import os

from langchain_core.runnables import (
    RunnableLambda, RunnableParallel,
    RunnablePassthrough, RunnableSequence
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Initialize the GROQ client
# Safe initialization fallback
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("CRITICAL: GROQ_API_KEY is missing! Check your .env file or environment variables.")

# client_groq = Groq(api_key=api_key)

# ── What is a Runnable? ───────────────────────────────────────────────────────
# Anything with .invoke(input) → output
# The pipe operator | creates a RunnableSequence
# Output of left becomes input of right

# ── Build a simple chain step by step ────────────────────────────────────────
# model  = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 1. Initialize the LangChain ChatGroq runnable wrapper
model = ChatGroq(
    model="llama-3.3-70b-versatile", # or whichever model you are using
    temperature=0,
    groq_api_key=api_key
)
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in one sentence for a beginner."
)
#prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
parser = StrOutputParser()

# The | operator: prompt | model | parser
# Equivalent to: parser.invoke(model.invoke(prompt.invoke({"topic": "..."})))
chain = prompt | model | parser

# .invoke — synchronous, returns final output
result = chain.invoke({"topic": "transformers"})
print("invoke() calling: Transformers")
print(result)
print("\n")


# .stream — yields tokens as they generate (crucial for UX)
print("stream() function: embeddings")
for chunk in chain.stream({"topic": "embeddings"}):
    print(chunk, end="", flush=True)
print("\n")

# .batch — parallel execution of multiple inputs
results = chain.batch([
    {"topic": "RAG"},
    {"topic": "fine-tuning"},
    {"topic": "quantization"},
], config={"max_concurrency": 3})  # run 3 in parallel

print("batch() results:")
for r in results:
    print(r)   
print("\n")

# .ainvoke — async (use in FastAPI, async servers)
import asyncio
result = asyncio.run(chain.ainvoke({"topic": "vector databases"}))
print("ainvoke() calling: vector databases")
print(result)
print("\n")

# ── RunnableParallel — run multiple chains simultaneously ─────────────────────
# Both branches run at the same time, results merged into a dict
parallel_chain = RunnableParallel({
    "simple":    prompt | model | parser,
    "technical": ChatPromptTemplate.from_template(
                     "Explain {topic} technically in one sentence."
                 ) | model | parser,
})
result = parallel_chain.invoke({"topic": "attention mechanism"})
print("RunnableParallel results:")
print("simple explanation: ")
print(result["simple"])
print("technical explanation: ")
print(result["technical"])

# ── RunnablePassthrough — pass input through unchanged ────────────────────────
# Critical pattern: include original input alongside transformed output
# Used constantly in RAG chains

chain_with_passthrough = RunnableParallel({
    "question": RunnablePassthrough(),   # passes question through unchanged
    "answer"  : prompt | model | parser, # also generates the answer
})
result = chain_with_passthrough.invoke({"topic": "HNSW"})
# result = {"question": {"topic": "HNSW"}, "answer": "HNSW is..."}
print("\n")
print("RunnablePassthrough results:")
print(result["question"])  # {'topic': 'HNSW'}
print(result["answer"])    # "HNSW is a graph-based algorithm for efficient nearest neighbor search."
print("\n")



# ── itemgetter — extract one key from a dict input ───────────────────────────
from operator import itemgetter
from langchain_core.vectorstores import InMemoryVectorStore

# 1. Mocking the required sub-components
# (Make sure you have OPENAI_API_KEY set in your environment variables)
# model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

# embeddings = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
parser = StrOutputParser()

# 2. Creating a minimalist In-Memory Retriever
# We feed it a list of dummy text data right on the spot
vector_store = InMemoryVectorStore.from_texts(
    texts=[
        "Enterprise plan costs $2000/month with unlimited API calls.",
        "Pro plan costs $299/month with 10k API calls/day."
    ],
    embedding=embeddings
)
# This perfectly initializes your missing variable!
your_retriever = vector_store.as_retriever(search_kwargs={"k": 1})

# ── RunnableLambda — wrap any Python function as a Runnable ──────────────────
def format_docs(docs: list) -> str:
    """Your custom logic, now composable in a chain."""
    return "\n\n".join([d.page_content for d in docs])

# Make it a Runnable so it can join the pipe
format_runnable = RunnableLambda(format_docs)

# CORRECT: Update the variables to match what your chain generates
promptNew = ChatPromptTemplate.from_template("""
Answer the question based only on the context provided.

Context: {context}

Question: {question}
Answer:""")

rag_chain = (
    RunnableParallel({
        "context" : itemgetter("question") | your_retriever | format_runnable,
        "question": itemgetter("question"),   # pass question through
    })
    | promptNew
    | model
    | parser
)

print("itemgetter() + your_retriever approach + RunnableLambda results:")
print("RAG chain results:")
result = rag_chain.invoke({"question": "How much is the pro plan?"})
print(result)
print("\n")

# ── Understanding what | actually does ───────────────────────────────────────
# a | b  is syntactic sugar for  a.__or__(b)
# which returns RunnableSequence([a, b])
# When you call .invoke(), it's just:
#   result = input
#   for step in steps:
#       result = step.invoke(result)
#   return result

# ── Configurable chains — swap components at runtime ─────────────────────────
from langchain_core.runnables import ConfigurableField
# from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

# configurable_model = ChatOpenAI(model="gpt-4o-mini").configurable_alternatives(
#     ConfigurableField(id="llm"),
#     default_key="gpt4o_mini",
#     claude=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
# )

configurable_model = ChatGroq(model="llama-3.3-70b-versatile").configurable_alternatives(
    ConfigurableField(id="llm"),
    default_key="llama_3_3",
    claude=ChatGroq(model="llama-3.3-70b-versatile"),
)


chain = prompt | configurable_model | parser

# Use default (GPT-4o-mini)
chain.invoke({"topic": "RAG"})

# Switch to Claude at runtime — no code change
chain.with_config(configurable={"llm": "claude"}).invoke({"topic": "RAG"})

print("Configurable chain results:")
print(chain.invoke({"topic": "RAG"}))
print("\n")


'''
____ OUTPUT _____________________________________

invoke() calling: Transformers
A transformer is a type of artificial intelligence model that uses self-attention mechanisms to process and understand sequential data, such as text or speech, by weighing the importance of different elements in relation to each other.


stream() function: embeddings
Embeddings are a way to represent complex data, like words or images, as simple numerical vectors in a high-dimensional space, allowing computers to understand and compare them more easily.

batch() results:
RAG (Retrieve, Augment, Generate) is an artificial intelligence model that combines retrieval of relevant information from a database with generation of text to create more accurate and informative responses to user queries.
Fine-tuning is a process in machine learning where a pre-trained model is adjusted and trained further on a smaller, specific dataset to improve its performance on a particular task, allowing it to learn and adapt to the new data.
Quantization is the process of converting a continuous signal, such as sound or light, into a digital format by representing it as a series of discrete numerical values, allowing it to be processed and stored by computers.


ainvoke() calling: vector databases
A vector database is a type of database that stores and manages complex data, such as images, text, and audio, as high-dimensional vectors, allowing for efficient similarity searches and machine learning-based applications.


RunnableParallel results:
simple explanation: 
The attention mechanism is a technique used in deep learning models that allows them to focus on specific parts of the input data that are most relevant to the task at hand, rather than processing the entire input equally.
technical explanation: 
The attention mechanism is a neural network component that computes a weighted sum of input elements, where the weights are learned based on the relevance of each element to the task at hand, allowing the model to focus on specific parts of the input data.


RunnablePassthrough results:
{'topic': 'HNSW'}
HNSW (Hierarchical Navigable Small World) is an efficient algorithm for finding the nearest neighbors in high-dimensional data by building a graph that connects similar data points, allowing for fast and accurate similarity searches.


Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 2499.96it/s]
itemgetter() + your_retriever approach + RunnableLambda results:
RAG chain results:
$299/month


Configurable chain results:
RAG (Retrieve, Augment, Generate) is a type of artificial intelligence model that combines retrieval of relevant information from a database with generation of text to create more accurate and informative responses.


'''