import streamlit as st
import pickle
import os

@st.cache_resource
def load_model(dir_checkpoint: str, model_name: str) -> dict:
    with open(os.path.join(dir_checkpoint, model_name), "rb") as file:
        model = pickle.load(file)
    return model

def generate_word(model: dict, prefix: str, top_k: int=5) -> list[str]:
    top_k_predict_words = None
    try:
        vocab = model[prefix]
        top_k_predict_words = sorted(vocab, key=vocab.get, reverse=True)[:top_k]
    except KeyError:
        pass
    return top_k_predict_words

def main(model: dict, init_text_input: str, top_k: int, n: int):
    if "init_text_input" not in st.session_state:
        st.session_state.init_text_input = ""

    text_input = st.text_input(
        label="🔍 Tìm kiếm (Vietnamese)",
        value=st.session_state.init_text_input,
    )

    prefix = tuple(word.lower() for word in text_input.split()[1-n:]) if text_input else None
    predict_words = generate_word(model, prefix, top_k)

    if not predict_words:
        if text_input:
            st.warning(f"⚠️ Không đủ thông tin để dự đoán !!!")
        else:
            st.warning("⚠️ Vui lòng nhập câu đầu vào !!!")
    else:
        cols = st.columns(top_k)
        for col, word in zip(cols, predict_words):
            with col:
                if st.button(word):
                    init_text_input = text_input + " " + word
                    st.session_state.init_text_input = init_text_input
                    st.rerun()

if __name__ == "__main__":
    st.title("n-grams Language Model for News")

    ngrams = st.selectbox(
        label="Giá trị của n-grams",
        options=[2, 3, 4, 5]
    )

    model = load_model(
        dir_checkpoint="checkpoint",
        model_name=f"{ngrams}grams.pkl"
    )
    main(model, "", top_k=5, n=ngrams)
