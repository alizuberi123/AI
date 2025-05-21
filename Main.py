import streamlit as st
import os
import shutil
from utils.rag_manager import list_rags, create_rag, get_rag_path

st.set_page_config(page_title="MIA", layout="centered")
st.title("📚 Welcome to MIA")

st.markdown("Create and manage your saved chats (RAGs).")

# --- Create New RAG ---
st.subheader("Start a New Chat")
new_rag = st.text_input("Enter a name for your new chat (RAG):")
if st.button("Create"):
    if new_rag.strip() == "":
        st.warning("❗ Please enter a name.")
    elif create_rag(new_rag):
        st.success(f"✅ RAG '{new_rag}' created.")
    else:
        st.warning("⚠️ A RAG with that name already exists.")

# --- View Existing RAGs ---
st.subheader("Your Chats (RAGs)")
rags = list_rags()

if not rags:
    st.info("No RAGs created yet.")
else:
    for rag in rags:
        col1, col2, col3 = st.columns([4, 2, 1])
        with col1:
            st.write(f"📁 **{rag}**")
        with col2:
            if st.button(f"Open", key=f"open_{rag}"):
                st.session_state.active_rag = rag
                st.switch_page("pages/ChatBot.py")
        with col3:
            if st.button("🗑️", key=f"delete_{rag}"):
                shutil.rmtree(get_rag_path(rag), ignore_errors=True)
                st.rerun()
