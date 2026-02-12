import numpy as np

USE_EMAILS = False  # set True to use email dataset

def load_texts():
    if USE_EMAILS:
        from emails import emails
        data = emails()

        if isinstance(data, (tuple, list)) and len(data) == 4:
            X_train, y_train, X_test, y_test = data
            texts = list(X_train) + list(X_test)
        elif isinstance(data, (tuple, list)) and len(data) == 2:
            X, y = data
            texts = list(X)
        elif isinstance(data, dict) and "train" in data:
            # assume list of (text,label)
            texts = [t for (t, l) in data["train"]]
        else:
            # assume list of (text,label)
            texts = [t for (t, l) in data]
        return texts[:3000]  

    # fallback: a tiny built-in corpus
    return [
        "I love frogs and want to learn more about them.",
        "Reminder: meeting tomorrow at 2pm in the lab.",
        "This is a spammy message about winning a prize, click now!",
        "I am working on graph neural networks and transformers.",
        "Boston is cold but the vibes are good.",
        "Please review the attached homework and submit by midnight.",
        "Protein structures can encode priors that help downstream tasks.",
    ]

def main():
    from sentence_transformers import SentenceTransformer

    texts = load_texts()
    print("loaded texts:", len(texts))

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # encode all texts
    emb = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print("\nType a query. Press enter on empty line to quit.")
    while True:
        query = input("\nquery> ").strip()
        if query == "":
            break

        q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        scores = emb @ q  # cosine sim because normalized
        topk = 5
        top_idx = np.argsort(-scores)[:topk]

        print("\nTop matches:")
        for rank, i in enumerate(top_idx, start=1):
            snippet = texts[i].replace("\n", " ")
            snippet = snippet[:160] + ("..." if len(snippet) > 160 else "")
            print(f"{rank}. score={scores[i]:.3f} | {snippet}")

if __name__ == "__main__":
    main()
