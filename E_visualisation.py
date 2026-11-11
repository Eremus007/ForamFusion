import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import umap


def show_curve(values, title="Loss"):
    """Plot the values given as an argument"""
    plt.plot(values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training {title} per epoch')
    plt.show()


def calculate_umap_features(embedding_sets):
    """
    embedding_sets: Dictionary containing embedding names and tensors.
    Example: {"DNA": dna_z, "Image": image_z}
    """
    embeddings = {
        name: emb.detach().cpu().numpy()
        for name, emb in embedding_sets.items()
    }
    combined = np.concatenate(list(embeddings.values()), axis=0)

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="cosine",
        random_state=42,
    )
    projected = reducer.fit_transform(combined)
    print(f"Projected shape: {projected.shape}")
    projected_sets = {}
    start = 0
    for name, embedding in embeddings.items():
        end = start + len(embedding)
        projected_sets[name] = projected[start:end]
        start = end

    return projected_sets


def show_umap_features(embedding_sets, labels):
    projected_sets = calculate_umap_features(embedding_sets)
    plot_data = []
    for name, projected in projected_sets.items():
        plot_df = pd.DataFrame({
            "x": projected[:, 0],
            "y": projected[:, 1],
            "label": labels,
            "modality": name
        })
        plot_data.append(plot_df)
    plot_df = pd.concat(plot_data, ignore_index=True)
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="label",
        symbol="modality",
        hover_data=["modality", "label"],
        title="Embedding Space (UMAP 2D Projection)",
        opacity=0.7,
    )
    fig.update_traces(marker=dict(size=5))
    fig.write_html("vector_space.html")
    fig.show()

if __name__ == "__main__":
    print("This script holds functions for embedding and learning curve visualisations, please run a different file.")