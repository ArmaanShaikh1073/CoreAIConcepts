# Decision framework: RAG vs alternatives

def should_use_rag(scenario: dict) -> str:
    """
    Before building a RAG system, run through this decision tree.
    Most over-engineering in AI comes from using RAG when you don't need it.
    """
    
    if scenario["knowledge_changes_daily"]:
        return "RAG — static fine-tuning can't keep up, need live retrieval"
    
    if scenario["corpus_size"] < 20 and scenario["changes_rarely"]:
        return "Prompt stuffing — just put the docs in the system prompt"
        # At <20 docs: retrieval overhead isn't worth it
        # Modern 128k context windows make this viable for small corpora
    
    if scenario["need_reasoning_over_own_knowledge"]:
        return "Pure LLM — RAG can't help with creative tasks, reasoning, coding help"
    
    if scenario["strict_latency_requirement_ms"] < 200:
        return "Consider prompt caching + fine-tuning instead of RAG retrieval"
        # RAG adds 100-500ms for embedding + search + reranking
    
    if scenario["corpus_size"] > 1_000_000 and scenario["very_structured"]:
        return "SQL/structured DB + LLM — vector search on tabular data is wrong tool"
    
    return "RAG — large, unstructured, changing knowledge corpus"

# Real scenarios
scenarios = [
    {"knowledge_changes_daily": False, "corpus_size": 10, "changes_rarely": True,
     "need_reasoning_over_own_knowledge": False, "strict_latency_requirement_ms": 500,
     "very_structured": False},   # → prompt stuffing
    
    {"knowledge_changes_daily": True, "corpus_size": 500_000, "changes_rarely": False,
     "need_reasoning_over_own_knowledge": False, "strict_latency_requirement_ms": 1000,
     "very_structured": False},   # → RAG
]

for s in scenarios:
    print(should_use_rag(s))


'''

_____ Output ______________________________________________

Prompt stuffing — just put the docs in the system prompt
RAG — static fine-tuning can't keep up, need live retrieval
'''