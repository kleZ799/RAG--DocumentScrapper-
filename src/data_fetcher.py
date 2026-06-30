"""
Utility: Download 60 Wikipedia articles as .txt files into ./data/
No API key needed. Run this once before main.py.
"""

import os
import time

def fetch_wikipedia_docs(output_dir: str = "./data", count: int = 60):
    try:
        import wikipediaapi
    except ImportError:
        print("Installing wikipedia-api...")
        os.system("pip install wikipedia-api -q")
        import wikipediaapi

    topics = [
        "Machine learning", "Deep learning", "Neural network",
        "Transformer (machine learning model)", "BERT (language model)",
        "GPT-3", "Attention mechanism", "Reinforcement learning",
        "Natural language processing", "Computer vision",
        "Knowledge graph", "Semantic search", "Vector database",
        "Word embedding", "Recurrent neural network",
        "Convolutional neural network", "Transfer learning",
        "Few-shot learning", "Zero-shot learning", "Prompt engineering",
        "Large language model", "Retrieval-augmented generation",
        "Information retrieval", "Question answering",
        "Named-entity recognition", "Sentiment analysis",
        "Text summarization", "Machine translation",
        "Speech recognition", "Object detection",
        "Generative adversarial network", "Variational autoencoder",
        "Backpropagation", "Gradient descent", "Overfitting",
        "Regularization (mathematics)", "Dropout (neural networks)",
        "Batch normalization", "Residual neural network",
        "Encoder–decoder architecture", "Self-supervised learning",
        "Contrastive learning", "Multimodal learning",
        "Graph neural network", "Knowledge distillation",
        "Federated learning", "Explainable artificial intelligence",
        "Bias in artificial intelligence", "AI safety",
        "Artificial general intelligence", "Turing test",
        "Support vector machine", "Random forest",
        "Decision tree", "K-nearest neighbors algorithm",
        "Principal component analysis", "Clustering",
        "Dimensionality reduction", "Feature engineering",
        "Cross-validation (statistics)", "Confusion matrix",
    ][:count]

    os.makedirs(output_dir, exist_ok=True)
    wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent='RAGMilestone1/1.0 (educational project)'
    )

    saved = 0
    for topic in topics:
        try:
            page = wiki.page(topic)
            if page.exists():
                filename = topic.replace(" ", "_").replace("/", "-") + ".txt"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"# {topic}\n\n")
                    f.write(page.text[:5000])  # first 5000 chars
                saved += 1
                print(f"✅ [{saved}/{count}] {topic}")
                time.sleep(0.3)  # be polite to Wikipedia
        except Exception as e:
            print(f"⚠️  Skipped '{topic}': {e}")

    print(f"\n🎉 Saved {saved} documents to '{output_dir}/'")
    return saved


if __name__ == "__main__":
    fetch_wikipedia_docs()
