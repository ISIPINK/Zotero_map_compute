from transformers import AutoTokenizer, AutoModel
import pandas as pd
import pickle


def get_embedding(row):
    tit = (
        row["Title"]
        + tokenizer.sep_token
        + "tags:" + str(row["Manual Tags"]) + str(row["Automatic Tags"])
        + tokenizer.sep_token
        + row["Abstract Note"]
    )
    inputs = tokenizer(tit, padding=True, truncation=True,
                       return_tensors="pt", max_length=512)
    result = model(**inputs)
    # take the first token in the batch as the embedding
    return result.last_hidden_state[:, 0, :].detach().numpy()[0]


def create_bib_specter_embedded():
    df = pd.read_csv("./Mijn Bibliotheek.csv")
    df = df.dropna(subset=["Title", "Abstract Note"])
    # took 3min for 300 items on my desktop no gpu ...
    print("start embedding")
    embeddings = []
    for index, row in df.iterrows():
        embeddings.append(get_embedding(row))

    with open('bib_specter_embedded.pkl', 'wb') as f:
        pickle.dump(embeddings, f)


if __name__ == "__main__":
    # load model and tokenizer
    # y global variable things
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    model = AutoModel.from_pretrained('allenai/specter')
    create_bib_specter_embedded()
