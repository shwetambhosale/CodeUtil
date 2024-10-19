import json
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory

def get_json_data(json_files):
    """Load JSON data from files and return as a list of dictionaries."""
    data = []  # This will hold all the JSON data
    for json_file in json_files:
        data.append(json.load(json_file))
    return data

def text_to_chunks(json_data):
    """Convert JSON content into chunks of text."""
    raw_text = json.dumps(json_data, indent=4)  # Convert JSON to readable text
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n",
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def simple_retriever(chunks, query):
    """A basic retriever that searches for a keyword or phrase in the chunks."""
    results = []
    print("Chunks available for retrieval:")  # Debugging output
    for chunk in chunks:
        print(chunk)  # Print each chunk for debugging
        if query.lower() in chunk.lower():
            results.append(chunk)
    return results


def get_conversation_chain(chunks):
    """Create a basic conversation-like retrieval system."""
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def conversation_chain(user_question):
        responses = simple_retriever(chunks, user_question)
        if responses:
            response = responses[0]  # Return the first relevant chunk found
            memory.save_context({"question": user_question}, {"answer": response})
        else:
            response = "No relevant information found."
            memory.save_context({"question": user_question}, {"answer": response})
        
        return {"chat_history": memory.load_memory_variables({"question": user_question})}

    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please click the 'Process' button to initialize the conversation.")
    else:
        # Check if the user input is "hi"
        if user_question.lower() == "hi":
            st.write(f"**Bot:** Hi, how can I help you?")
            return
        
        # Otherwise, proceed with the normal conversation flow
        conversation_chain = st.session_state.conversation
        response = conversation_chain(user_question)
        
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history['chat_history']):
            if i % 2 == 0:
                st.write(f"**User:** {message.content}")
            else:
                try:
                    formatted_json = json.loads(message.content)
                    st.json(formatted_json)  
                except json.JSONDecodeError:
                    st.write(f"**Bot:** {message.content}")


def main():
    st.set_page_config(page_title="Chat with JSON", page_icon=':file_folder:')
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with JSON File :file_folder:")
    
    with st.sidebar:
        st.subheader("Your JSON Files")
        json_files = st.file_uploader("Upload JSON Files", type=['json'], accept_multiple_files=True)
        
        if st.button("Process"):
            if json_files:
                with st.spinner("Processing..."):
                    # Load JSON data
                    json_data = get_json_data(json_files)
                    
                    # Split the JSON content into text chunks
                    chunks = text_to_chunks(json_data)
                    
                    # Create a conversation chain for retrieval
                    st.session_state.conversation = get_conversation_chain(chunks)

    user_question = st.text_input("Ask a question about the JSON data: ")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()
