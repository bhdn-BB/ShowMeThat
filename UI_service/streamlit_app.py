import streamlit as st

if __name__ == "__main__":

    st.set_page_config(
        page_title="ShowMeThat 🔍",
        page_icon="🎥",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.title("🎬 ShowMeThat")
    st.markdown("**Find the right moment in a YouTube video with the help of AI!**")

    st.divider()

    search_type = st.radio(
        "Choose your search type:",
        ("🔍 By Text", "🖼️ By Image"),
        horizontal=True
    )

    match search_type:
        case "🔍 By Text":
            text_input = st.text_input(
                "Enter a text query to search:",
                placeholder="For example: 'Person on a scooter'"
            )
        case "🖼️ By Image":
            image_file = st.file_uploader(
                "Upload an image to find matching video segments:",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=False
            )
        case _:
            st.warning("Error: Please select a search type.")