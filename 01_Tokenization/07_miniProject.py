# mini_project_token_chunker.py
# Production-grade text chunker that respects token boundaries
# This is used in every serious RAG pipeline

import tiktoken
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    token_count: int
    chunk_index: int
    overlap_tokens: int

class TokenAwareChunker:
    """
    Splits text into chunks with exact token counts and configurable overlap.
    
    Why this matters: if you chunk by characters (e.g. 1000 chars), 
    some chunks will have 200 tokens and others 400+ depending on word length.
    Your embedding model has a hard token limit (e.g. 512 for most models).
    Silent truncation = lost information = worse retrieval.
    """
    
    def __init__(
        self, 
        model: str = "gpt-4",
        chunk_size: int = 512,    # tokens per chunk
        overlap: int = 50,        # overlap tokens between chunks
    ):
        self.enc = tiktoken.encoding_for_model(model)
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))
    
    def chunk(self, text: str) -> list[Chunk]:
        """Split text into token-bounded chunks with overlap."""
        tokens = self.enc.encode(text)
        total_tokens = len(tokens)
        chunks = []
        
        start = 0
        chunk_idx = 0
        
        while start < total_tokens:
            end = min(start + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            
            # Decode back to text
            chunk_text = self.enc.decode(chunk_tokens)
            
            chunks.append(Chunk(
                text=chunk_text,
                token_count=len(chunk_tokens),
                chunk_index=chunk_idx,
                overlap_tokens=self.overlap if chunk_idx > 0 else 0,
            ))
            
            # Move start forward, but keep overlap
            start = end - self.overlap
            chunk_idx += 1
            
            if end == total_tokens:
                break
        
        return chunks
    
    def chunk_sentences(self, text: str) -> list[Chunk]:
        """
        Better approach: chunk at sentence boundaries, 
        respecting token limits. Avoids cutting mid-sentence.
        """
        import re
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_tokens = []
        current_text_parts = []
        chunk_idx = 0
        
        for sentence in sentences:
            sent_tokens = self.enc.encode(sentence)
            
            # If adding this sentence would exceed limit, save current chunk
            if (len(current_tokens) + len(sent_tokens) > self.chunk_size 
                    and current_tokens):
                
                chunk_text = ' '.join(current_text_parts)
                # meaning of this code line
                # if current_text_parts = ['Hello', 'world', 'this', 'is', 'AI']
                # then chunk_text = 'Hello world this is AI'
                # ' '.join()' concatenates list of strings into one string with spaces in between

                chunks.append(Chunk(
                    text=chunk_text,
                    token_count=len(current_tokens),
                    chunk_index=chunk_idx,
                    overlap_tokens=0,
                ))
                chunk_idx += 1
                
                # Start new chunk with overlap from end of current
                overlap_text = current_text_parts[-2:] if len(current_text_parts) >= 2 else current_text_parts
                # this line means that we take the last 2 sentences from current_text_parts to use as overlap for the next chunk

                current_text_parts = overlap_text + [sentence]
                # this line means that we create a new list of sentences for the next chunk, starting with the overlap sentences and adding the current sentence

                current_tokens = self.enc.encode(' '.join(current_text_parts))
            else:
                current_text_parts.append(sentence)
                current_tokens.extend(sent_tokens)
        
        # Don't forget the last chunk
        if current_text_parts:
            chunks.append(Chunk(
                text=' '.join(current_text_parts),
                token_count=len(current_tokens),
                chunk_index=chunk_idx,
                overlap_tokens=0,
            ))
        
        return chunks

# ── Utility: validate chunks fit your embedding model ────────────────────────
def validate_chunks_for_model(chunks: list[Chunk], max_tokens: int = 512):
    """Always run this before sending chunks to embedding API."""
    violations = [c for c in chunks if c.token_count > max_tokens]
    if violations:
        print(f"WARNING: {len(violations)} chunks exceed {max_tokens} token limit!")
        for v in violations:
            print(f"  Chunk {v.chunk_index}: {v.token_count} tokens")
    else:
        print(f"All {len(chunks)} chunks within {max_tokens} token limit.")
    return len(violations) == 0

# ── Test it ──────────────────────────────────────────────────────────────────
sample_text = """
Artificial intelligence is transforming the world at an unprecedented pace. 
Machine learning models can now understand natural language, generate images, 
and write code. The field of NLP has especially advanced, with transformer 
architectures enabling models like GPT and BERT to achieve human-level 
performance on many tasks. Tokenization is a fundamental step in all of 
these pipelines. Without proper tokenization, models cannot process text 
at all. The BPE algorithm, invented in 1994 for data compression and 
adapted for NLP in 2015, is now the backbone of most modern LLMs.
""".strip()

chunker = TokenAwareChunker(chunk_size=60, overlap=10)

print("=== Raw token chunking ===")
chunks = chunker.chunk(sample_text)
for c in chunks:
    print(f"Chunk {c.chunk_index}: {c.token_count} tokens | '{c.text}...'")
    
validate_chunks_for_model(chunks, max_tokens=60)

print("\n=== Sentence-boundary chunking ===")
chunks = chunker.chunk_sentences(sample_text)
for c in chunks:
    print(f"Chunk {c.chunk_index}: {c.token_count} tokens | '{c.text}...'")
    
validate_chunks_for_model(chunks, max_tokens=60)


'''
___ output ________________________________________________________________________

=== Raw token chunking ===
Chunk 0: 60 tokens | 'Artificial intelligence is transforming the world at an unprecedented pace. 
Machine learning models can now understand natural language, generate images, 
and write code. The field of NLP has especially advanced, with transformer 
architectures enabling models like GPT and BERT to achieve human-level 
performance on...'
Chunk 1: 60 tokens | ' and BERT to achieve human-level 
performance on many tasks. Tokenization is a fundamental step in all of 
these pipelines. Without proper tokenization, models cannot process text 
at all. The BPE algorithm, invented in 1994 for data compression and 
adapted for NLP in...'
Chunk 2: 25 tokens | ' data compression and 
adapted for NLP in 2015, is now the backbone of most modern LLMs....'

All 3 chunks within 60 token limit.

=== Sentence-boundary chunking ===
Chunk 0: 29 tokens | 'Artificial intelligence is transforming the world at an unprecedented pace. Machine learning models can now understand natural language, generate images, 
and write code....'
Chunk 1: 62 tokens | 'Artificial intelligence is transforming the world at an unprecedented pace. Machine learning models can now understand natural language, generate images, 
and write code. The field of NLP has especially advanced, with transformer 
architectures enabling models like GPT and BERT to achieve human-level 
performance on many tasks....'
Chunk 2: 63 tokens | 'Machine learning models can now understand natural language, generate images, 
and write code. The field of NLP has especially advanced, with transformer 
architectures enabling models like GPT and BERT to achieve human-level 
performance on many tasks. Tokenization is a fundamental step in all of 
these pipelines....'
Chunk 3: 59 tokens | 'The field of NLP has especially advanced, with transformer 
architectures enabling models like GPT and BERT to achieve human-level 
performance on many tasks. Tokenization is a fundamental step in all of 
these pipelines. Without proper tokenization, models cannot process text 
at all....'
Chunk 4: 62 tokens | 'Tokenization is a fundamental step in all of 
these pipelines. Without proper tokenization, models cannot process text 
at all. The BPE algorithm, invented in 1994 for data compression and 
adapted for NLP in 2015, is now the backbone of most modern LLMs....'

WARNING: 3 chunks exceed 60 token limit!
  Chunk 1: 62 tokens
  Chunk 2: 63 tokens
  Chunk 4: 62 tokens

'''