import streamlit as st
import requests
import json



backend_url = "http://localhost:8000"

if "page" not in st.session_state: st.session_state["page"] = "session"
if "session_id" not in st.session_state: st.session_state["session_id"] = ""
if "uploaded_files" not in st.session_state: st.session_state["uploaded_files"] = []
if "active_files" not in st.session_state: st.session_state["active_files"] = []
if "suggestions" not in st.session_state: st.session_state["suggestions"] = []
if "query_text" not in st.session_state: st.session_state["query_text"] = ""
if "last_mcq" not in st.session_state: st.session_state["last_mcq"] = []

# PAGE 1: Session creation
if st.session_state["page"] == "session":
    st.header("Create a Session")
    user_id = st.text_input("User ID")
    session_name = st.text_input("Session Name")
    if st.button("Create Session"):
        payload = {"user_id": user_id, "session_name": session_name}
        resp = requests.post(f"{backend_url}/api/sessions/create", json=payload)
        if resp.ok:
            data = resp.json()
            st.session_state["session_id"] = data["session_id"]
            st.success(f"Created! ID: {data['session_id']}")
            st.session_state["page"] = "workspace"
            st.rerun()
        else:
            st.error("Session creation failed.")

# PAGE 2: Workspace
elif st.session_state["page"] == "workspace":
    st.sidebar.header("Sources")
    st.sidebar.write("Upload files and select sources for context below.")

    # File upload
    uploaded = st.sidebar.file_uploader("Add sources", accept_multiple_files=True)
    if st.sidebar.button("Upload Sources"):
        if uploaded:
            files = [("files", (f.name, f, f.type)) for f in uploaded]
            data = {"session_id": st.session_state["session_id"]}
            resp = requests.post(f"{backend_url}/api/files/upload", files=files, data=data)
            if resp.ok:
                result = resp.json()
                new_files = [f["filename"] for f in result["results"] if f["status"] == "accepted"]
                st.success("Files uploaded.")
                st.session_state["uploaded_files"].extend(new_files)
                st.session_state["active_files"] = st.session_state["uploaded_files"][:]
                st.sidebar.write("Current sources:")
                st.sidebar.write(st.session_state["uploaded_files"])
                st.session_state["suggestions"] = result.get("suggestions", [])
                st.rerun()
            else:
                st.error("Upload failed.")

    # Delete file option
    st.sidebar.markdown("---")
    st.sidebar.subheader("Delete File")
    if st.session_state["uploaded_files"]:
        file_to_delete = st.sidebar.selectbox(
            "Select a file to delete",
            st.session_state["uploaded_files"],
            key="delete_file_select"
        )
        if st.sidebar.button("Delete Selected File"):
            payload = {
                "session_id": st.session_state["session_id"],
                "filename": file_to_delete
            }
            resp = requests.post(f"{backend_url}/api/files/delete", json=payload)
            if resp.ok:
                st.session_state["uploaded_files"].remove(file_to_delete)
                if file_to_delete in st.session_state["active_files"]:
                    st.session_state["active_files"].remove(file_to_delete)
                st.sidebar.success(f"Deleted {file_to_delete}")
                st.rerun()
            else:
                st.sidebar.error("Delete failed.")
    else:
        st.sidebar.write("No files uploaded.")

    # Multiselect for file filtering (keeps selection in session state)
    st.session_state["active_files"] = st.sidebar.multiselect(
        "Select sources for Query/MCQ:",
        st.session_state["uploaded_files"],
        default=st.session_state["active_files"]
    )

    # Change session button
    if st.sidebar.button("Change Session"):
        st.session_state["page"] = "session"
        st.session_state["session_id"] = ""
        st.session_state["uploaded_files"] = []
        st.session_state["active_files"] = []
        st.session_state["suggestions"] = []
        st.session_state["query_text"] = ""
        st.session_state["last_mcq"] = []
        st.rerun()

    st.header("Workspace")
    tab1, tab2 = st.tabs(["Query", "MCQ"])

    # --- Query tab ---
    with tab1:
        st.subheader("Ask a question")
        cur_query_text = st.text_area("Enter your question", value=st.session_state.get("query_text", ""))
        query_btn = st.button("Run Query", key="run_query_btn")

        # --- Show latest answer first ---
        if st.session_state.get("last_answer"):
            st.markdown(f"**Answer:** {st.session_state['last_answer']}")
        if st.session_state.get("last_sources"):
            st.write("**Sources:**", st.session_state["last_sources"])

        # --- Suggestions come after answer ---
        st.subheader("Suggestions")
        suggestion_clicked = None
        for idx, suggestion in enumerate(st.session_state["suggestions"]):
            if st.button(suggestion, key=f"sugg_btn_{idx}"):
                suggestion_clicked = suggestion
                break

        if suggestion_clicked is not None:
            st.session_state["query_text"] = suggestion_clicked
            st.rerun()

        # Query logic for new answer
        if query_btn and cur_query_text.strip():
            payload = {
                "session_id": st.session_state["session_id"],
                "query": cur_query_text,
                "files": st.session_state["active_files"]
            }
            resp = requests.post(f"{backend_url}/api/qa/query", json=payload)
            if resp.ok:
                data = resp.json()
                st.session_state["last_answer"] = data.get("answer")
                st.session_state["last_sources"] = data.get("sourcedocs") or data.get("sources")
                st.session_state["suggestions"] = data.get("suggestions", [])
                st.session_state["query_text"] = cur_query_text
                st.rerun()
            else:
                st.error("Query failed.")


    # --- MCQ tab ---
    with tab2:
        st.subheader("Generate MCQs")
        num_questions = st.number_input("Number of MCQs", min_value=1, max_value=20, value=5)
        mcq_response = None
        if st.button("Generate MCQs", key="mcq_btn"):
            payload = {
                "session_id": st.session_state["session_id"],
                "num_questions": int(num_questions),
                "files": st.session_state["active_files"]
            }
            resp = requests.post(f"{backend_url}/api/qa/generate-mcq", json=payload)
            if resp.ok:
                mcq_response = resp.json().get("mcqs")
                st.session_state["last_mcq"] = mcq_response
            else:
                st.error("MCQ generation failed.")

        mcq_to_show = st.session_state.get("last_mcq", [])
        if mcq_to_show:
            st.markdown("### Multiple Choice Questions")
            for idx, mcq in enumerate(mcq_to_show):
                st.markdown(f"**Q{idx+1}. {mcq['question']}**")
                options = list(mcq["choices"].items())
                user_choice = st.radio(
                    "Select one:",
                    options=[f"{label}: {text}" for label, text in options],
                    key=f"mcq_{idx}_opt"
                )
                with st.expander("Show Answer"):
                    correct = mcq.get("correct_answer", "")
                    st.success(f"Correct Answer: {correct}: {mcq['choices'][correct]}")
                st.markdown("---")
