import streamlit as st
import pickle
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Toxic Comment Filter")

st.title("📱 Social Media Toxic Comment Filter")

# 🔥 Highlight toxic words (UPDATED LIST)
def highlight_toxic_words(text):
    toxic_words = [
        "hate", "stupid", "idiot", "shut up",
        "dumb", "worst", "nonsense", "annoying"
    ]

    words = text.split()
    result = []

    for w in words:
        if w.lower() in toxic_words:
            result.append(f"🔴 {w}")
        else:
            result.append(w)

    return " ".join(result)

# 🔥 Emotion detection
def detect_emotion(text):
    text = text.lower()

    if any(word in text for word in ["hate", "stupid", "idiot", "dumb"]):
        return "😡 Angry"
    elif any(word in text for word in ["love", "amazing", "great"]):
        return "😊 Happy"
    elif any(word in text for word in ["sad", "bad"]):
        return "😢 Sad"
    else:
        return "😐 Neutral"


comments = st.text_area("Paste comments (one per line):")

if st.button("Analyze"):
    comment_list = comments.split("\n")

    st.markdown("## 💬 Comments Feed")

    # 🔥 Counters for dashboard
    toxic_count = 0
    non_toxic_count = 0

    for i, c in enumerate(comment_list):
        if c.strip() != "":
            data = vectorizer.transform([c])
            pred = model.predict(data)[0]
            prob = model.predict_proba(data)[0][1]

            st.markdown(f"**👤 User{i+1}**")

            # Original comment
            st.write(c)

            # Highlighted text
            st.markdown("**Highlighted:**")
            st.write(highlight_toxic_words(c))

            # Emotion detection
            emotion = detect_emotion(c)
            st.info(f"Emotion: {emotion}")

            # Toxic prediction + count
            if pred == 1:
                toxic_count += 1
                st.error(f"😡 Toxic ({prob*100:.1f}%)")
            else:
                non_toxic_count += 1
                st.success("🙂 Not Toxic")

            st.markdown("---")

    # 🔥 Dashboard
    st.markdown("## 📊 Summary Dashboard")

    labels = ['Toxic', 'Non-Toxic']
    values = [toxic_count, non_toxic_count]

    fig, ax = plt.subplots()
    ax.bar(labels, values)

    st.pyplot(fig)
