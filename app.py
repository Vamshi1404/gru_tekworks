import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="GRU Shakespeare Generator",
    page_icon="📜"
)

model = tf.keras.models.load_model(
    "models/gru_shakespeare.keras"
)

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("models/max_sequence_len.pkl", "rb") as f:
    max_sequence_len = pickle.load(f)

st.title("📜 Shakespeare Text Generator (GRU)")

seed_text = st.text_input(
    "Starting Text",
    "to be"
)

num_words = st.slider(
    "Words to Generate",
    5,
    50,
    20
)

if st.button("Generate"):

    generated_text = seed_text

    for _ in range(num_words):

        token_list = tokenizer.texts_to_sequences(
            [generated_text]
        )[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_len - 1,
            padding="pre"
        )

        prediction = model.predict(
            token_list,
            verbose=0
        )

        predicted_index = np.argmax(
            prediction,
            axis=-1
        )[0]

        output_word = ""

        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break

        generated_text += " " + output_word

    st.subheader("Generated Text")
    st.write(generated_text)