# prompts_deep_dive.py

from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ── Basic ChatPromptTemplate ──────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Answer in {language}."),
    ("human",  "{question}"),
])

# Variables are filled at invoke time
messages = prompt.invoke({
    "role":     "senior Python engineer",
    "language": "English",
    "question": "What is a generator?",
})
print(messages)  # list of BaseMessage objects ready for the model

# ── MessagesPlaceholder — inject conversation history ────────────────────────
# This is how you build chatbots with memory
prompt_with_history = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),  # ← history injected here
    ("human", "{question}"),
])

# history is a list of HumanMessage and AIMessage objects
history = [
    HumanMessage(content="My name is Arjun."),
    AIMessage(content="Nice to meet you, Arjun!"),
]

messages = prompt_with_history.invoke({
    "history":  history,
    "question": "What's my name?",
})
# The model will see: system → human("My name is...") → AI("Nice to...") → human("What's my name?")

# ── Few-shot prompting ────────────────────────────────────────────────────────
examples = [
    {"input": "happy",   "output": "sad"},
    {"input": "tall",    "output": "short"},
    {"input": "bright",  "output": "dark"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai",    "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt = example_prompt,
    examples       = examples,
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "Give the antonym of every input word."),
    few_shot_prompt,
    ("human", "{word}"),
])

# ── Partial prompts — pre-fill some variables ─────────────────────────────────
# Useful when some variables are known at setup, others at query time
base_prompt = ChatPromptTemplate.from_template(
    "Today is {date}. Answer this question: {question}"
)
from datetime import datetime
prompt_with_date = base_prompt.partial(date=datetime.now().strftime("%Y-%m-%d"))

# Now only needs {question} at invoke time
prompt_with_date.invoke({"question": "What day is it?"})


'''
____ OUTPUT _____________________________________

messages=[SystemMessage(content='You are a senior Python engineer. Answer in English.', additional_kwargs={}, response_metadata={}), HumanMessage(content='What is a generator?', additional_kwargs={}, response_metadata={})]
'''