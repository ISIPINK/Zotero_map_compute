# runs on my computer for 2 mins
import umap
import pacmap
import pickle

if __name__ == "__main__":
    with open('bib_specter_embedded.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    word_map = umap.UMAP(
        metric="cosine", random_state=42).fit_transform(embeddings)
    pp2 = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=5,
        MN_ratio=0.5,
        FP_ratio=2.0,
        distance="angular",
        random_state=42)

    pp3 = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=7,
        MN_ratio=0.5,
        FP_ratio=2.0,
        distance="angular",
        random_state=42)

    word_map2 = pp2.fit_transform(embeddings, init="pca")
    word_map3 = pp3.fit_transform(embeddings, init="pca")

    with open('word_umap.pkl', 'wb') as f:
        pickle.dump(word_map, f)
    with open('word_pacmap2.pkl', 'wb') as f:
        pickle.dump(word_map2, f)
    with open('word_pacmap3.pkl', 'wb') as f:
        pickle.dump(word_map3, f)
