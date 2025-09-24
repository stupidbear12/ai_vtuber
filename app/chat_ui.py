import streamlit as st
from llm.infer import load_model_and_tokenizer, generate

st.set_page_config(page_title="AI VTuber Chat", page_icon="🤖")
st.title("AI VTuber Chat 🤖")

@st.cache_resource
def get_model_and_tok():
    return load_model_and_tokenizer()

model, tok = get_model_and_tok()


col1, col2 = st.columns([1, 6])
with col1:
    if st.button("🧹 새 대화"):
        st.session_state.history = []


if "history" not in st.session_state:
    st.session_state.history = []  


for role, msg in st.session_state.history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(msg)


user_msg = st.chat_input("메시지를 입력하세요...")
if user_msg:
    st.session_state.history.append(("user", user_msg))
    with st.chat_message("user"):
        st.markdown(user_msg)

  
    turns = st.session_state.history[-5:]
    context = ""
    for role, msg in turns:
        prefix = "User: " if role == "user" else "Bot: "
        context += prefix + msg + "\n"
    if not context.endswith("Bot: "):
        context += "Bot: "  

    
    reply = generate(
        model, tok, context,
        max_new_tokens=120, temperature=0.9, top_k=50
    )

    st.session_state.history.append(("assistant", reply))
    with st.chat_message("assistant"):
        st.markdown(reply)

    

