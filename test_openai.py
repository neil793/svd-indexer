from openai import OpenAI

client = OpenAI()

def test_embeddings(model: str):
    print(f"\nTesting model: {model}")
    try:
        response = client.embeddings.create(
            model=model,
            input="Quick access test for embeddings."
        )
        emb = response.data[0].embedding
        print("✅ Success")
        print(f"Embedding length: {len(emb)}")
    except Exception as e:
        print("❌ Failed")
        print(e)

if __name__ == "__main__":
    # Try both – one of these is guaranteed to exist if embeddings are enabled
    test_embeddings("text-embedding-3-small")
    test_embeddings("text-embedding-3-large")
