import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
import gradio as gr
DATA_FILE = r"C:\Users\KIIT\Desktop\Global AI\fine mast data.csv"
KMER_SIZE = 2
TOP_N = 3
SIM_THRESHOLD = 0.3
df = pd.read_csv(DATA_FILE)

seq_cols = df.columns[1:6]
long_df = df.melt(id_vars=[df.columns[0]], value_vars=seq_cols,
                  var_name="source_col", value_name="sequence")
long_df.dropna(subset=["sequence"], inplace=True)
long_df["sequence"] = long_df["sequence"].astype(str).str.strip()
long_df = long_df[long_df["sequence"] != ""]
def kmer_split(seq, k=2):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)] if len(seq) >= k else [seq]
vectorizer = CountVectorizer(analyzer=lambda x: kmer_split(x, KMER_SIZE))
X = vectorizer.fit_transform(long_df["sequence"])
nn = NearestNeighbors(n_neighbors=TOP_N, metric="cosine")
nn.fit(X)

def predict_sequence(seq):
    vec = vectorizer.transform([seq])
    distances, indices = nn.kneighbors(vec)
    similarities = 1 - distances[0]

    results = []
    for idx, sim in zip(indices[0], similarities):
        results.append({
            "ID": long_df.iloc[idx, 0],
            "Similarity": round(float(sim), 3)
        })
    if results[0]["Similarity"] < SIM_THRESHOLD:
        return "Unknown", results

    return results[0]["ID"], results

def gradio_predict(user_seq):
    best_id, matches = predict_sequence(user_seq.strip())
    matches_str = "\n".join([f"{m['ID']}  | Similarity: {m['Similarity']}" for m in matches])
    return best_id, matches_str

iface = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(lines=2, label="Enter Protein Sequence"),
    outputs=[gr.Textbox(label="Best Match ID"), gr.Textbox(label="Top Matches")],
    title="Protein Sequence Matcher",
    description="Input a protein sequence and get the closest matching sequence IDs and similarity scores."
)

if __name__ == "__main__":
    iface.launch()
