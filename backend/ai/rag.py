from config import settings

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(settings.embedder_model)
    return _embedder

#convert a to chunks
def chunky(a, maxi):
    chunks = []
    mini = 0
    while mini < len(a):
        t = a[mini:maxi]
        chunks.append(t)
        mini = mini + 100
        maxi = maxi + 100
    return chunks

#chunks to vectors
def embed(a):
    embedder = get_embedder()
    embeddings = embedder.encode(
        a,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")
    return embeddings

async def rag_search(query: str, embeddings, chunks, top_k=3 ):
    import faiss

    rag_context = ""

    if len(embeddings) < top_k :
        for chunk in chunks:
            rag_context += chunk
        return rag_context

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    query_embedding = embed([query])
    scores, ids = index.search(query_embedding, top_k)

    results = []
    for k in range(top_k):
        results.append({
            "rank": k + 1,
            "text": chunks[ids[0][k]],
            "score": float(scores[0][k])
        })

    for x in results :
        rag_context += x["text"]
    return rag_context






