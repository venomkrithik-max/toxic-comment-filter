import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Toxic Comment Filter")

st.title("📱 Social Media Toxic Comment Filter")

comments = st.text_area("Paste comments (one per line):")

if st.button("Analyze"):
    comment_list = comments.split("\n")

    st.markdown("## 💬 Comments Feed")

    for i, c in enumerate(comment_list):
        if c.strip() != "":
            data = vectorizer.transform([c])
            pred = model.predict(data)[0]
            prob = model.predict_proba(data)[0][1]

            st.markdown(f"**👤 User{i+1}**")
            st.write(c)

            if pred == 1:
                st.error(f"😡 Toxic ({prob*100:.1f}%)")
            else:
                st.success("🙂 Not Toxic")

            st.markdown("---")
