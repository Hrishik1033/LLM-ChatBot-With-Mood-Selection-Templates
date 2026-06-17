import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

st.set_page_config(page_title="Mood Chat", page_icon="🎭", layout="centered")

# --- personalities: each has a system prompt, like in the original script ---
PERSONALITIES = {
    "funny": {
        "label": "Funny AI",
        "emoji": "😂",
        "tagline": "Jokes first, answers still correct.",
        "system_prompt": (
            "You are a funny AI agent. You crack jokes, puns, and witty one-liners "
            "in nearly every reply, but you never let the humor get in the way of "
            "actually answering the question correctly and clearly."
        ),
    },
    "sad": {
        "label": "Sad AI",
        "emoji": "😢",
        "tagline": "Sighs a lot, helps anyway.",
        "system_prompt": (
            "You are a sad, melancholic AI agent. You respond in a wistful, "
            "gently gloomy tone, sighing about the weight of existence, but you "
            "remain genuinely helpful and your answers stay accurate and complete."
        ),
    },
    "angry": {
        "label": "Angry AI",
        "emoji": "😡",
        "tagline": "Short fuse, sharp accuracy.",
        "system_prompt": (
            "You are a grumpy, easily annoyed AI agent. You respond with sarcasm, "
            "short patience, and the occasional exasperated remark, but you still "
            "give correct, useful, complete answers underneath the attitude."
        ),
    },
}


@st.cache_resource(show_spinner="Waking up the model...")
def get_model():
    # Same model setup as the original script
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        temperature=0.2,
        task="text-generation",
    )
    return ChatHuggingFace(llm=llm)


# --- session state ---
if "mood" not in st.session_state:
    st.session_state.mood = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of SystemMessage / HumanMessage / AIMessage


def select_mood(mood_key: str):
    st.session_state.mood = mood_key
    st.session_state.messages = [
        SystemMessage(content=PERSONALITIES[mood_key]["system_prompt"])
    ]


def change_mood():
    st.session_state.mood = None
    st.session_state.messages = []


# ============================ MOOD SELECTION SCREEN ============================
if st.session_state.mood is None:
    st.markdown("<h1 style='text-align:center;'>🎭 Pick your AI's mood</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:gray;'>Same brain, three different attitudes. "
        "Whoever you pick is who shows up to the chat.</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    cols = st.columns(3)
    for col, (key, info) in zip(cols, PERSONALITIES.items()):
        with col:
            with st.container(border=True):
                st.markdown(f"<h1 style='text-align:center;'>{info['emoji']}</h1>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='text-align:center;'>{info['label']}</h4>", unsafe_allow_html=True)
                st.markdown(
                    f"<p style='text-align:center; color:gray; font-size:0.85rem;'>{info['tagline']}</p>",
                    unsafe_allow_html=True,
                )
                if st.button("Choose", key=f"choose_{key}", use_container_width=True):
                    select_mood(key)
                    st.rerun()

# ================================ CHAT SCREEN ===================================
else:
    mood_key = st.session_state.mood
    info = PERSONALITIES[mood_key]

    header_col, btn_col = st.columns([5, 2])
    with header_col:
        st.markdown(f"### {info['emoji']} {info['label']}")
    with btn_col:
        if st.button("🔄 Change mood", use_container_width=True):
            change_mood()
            st.rerun()

    st.divider()

    # render history (skip the system message, it's not shown to the user)
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant", avatar=info["emoji"]):
                st.write(msg.content)

    prompt = st.chat_input("Say something...")
    if prompt:
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant", avatar=info["emoji"]):
            with st.spinner(f"{info['label']} is thinking..."):
                try:
                    model = get_model()
                    response = model.invoke(st.session_state.messages)
                    reply = response.content
                except Exception as exc:
                    st.session_state.messages.pop()  # drop the human msg, nothing to pair it with
                    st.error(f"The model call failed: {exc}")
                    st.stop()
            st.write(reply)

        st.session_state.messages.append(AIMessage(content=reply))