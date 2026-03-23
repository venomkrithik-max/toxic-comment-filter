import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Toxic Comment Filter")

st.title("📱 Social Media Toxic Comment Filter")

# 🔥 Highlight function (NEW)
def highlight_toxic_words(text):
    toxic_words = ["hate", "stupid", "idiot", "shut up"]
    words = text.split()
    result = []

    for w in words:
        if w.lower() in toxic_words:
            result.append(f"🔴 {w}")
        else:
            result.append(w)

    return " ".join(result)


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

            # Original text
            st.write(c)

            # 🔥 Highlighted text (NEW)
            st.markdown("**Highlighted:**")
            st.write(highlight_toxic_words(c))

            if pred == 1:
                st.error(f"😡 Toxic ({prob*100:.1f}%)")
            else:
                st.success("🙂 Not Toxic")

            st.markdown("---")
