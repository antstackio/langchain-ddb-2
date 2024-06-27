import streamlit as st
import requests
import json
import boto3

st.set_page_config(page_title="Streamlit App for Streaming Response", layout="wide")
st.header("Serverless RAG Application")
st.sidebar.header("Parameters")

url = st.sidebar.text_input("Lambda URL:", "")
user = st.sidebar.text_input("User:", "user3")
session = st.sidebar.text_input("Session:", "session2")
openai_api_key = st.sidebar.text_input("OpenAI API key:")
langchain_api_key = st.sidebar.text_input("Langchain API key:")
pinecone_api_key = st.sidebar.text_input("Pinecone API key:", "")
pinecone_index = st.sidebar.text_input("Pinecone Index:", "car-data")
question = st.text_input("Message:", "Tell me about Kia Seltos")
k_value = st.number_input("K value:", 5)

payload = {
    "question": question,
    "user": user,
    "session": session,
    "langchain_api_key": langchain_api_key,
    "openai_api_key": openai_api_key,
    "pinecone_api_key": pinecone_api_key,
    "pinecone_index": pinecone_index,
    "k_value": k_value
}

# Button to send a request
if st.button("Send Request"):
    # Function to stream data
    def stream_data(url, data):
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, data=json.dumps(data), headers=headers, stream=True)     

        for line in response.iter_lines():
            if line:
                # Decode the line and yield it
                yield line.decode('utf-8')

    # Create a placeholder for the streaming response
    response_placeholder = st.empty()
    response_content = ""
    # Stream the data and print it live
    for response_line in stream_data(url, payload):
        # Accumulate the response content
        response_content += response_line + "\n"
        # Update the placeholder with the current response content formatted as Markdown
        response_placeholder.markdown(response_content)

    # Retrieve chat history from DDB
    dynamodb = boto3.client('dynamodb')
    response = dynamodb.get_item(TableName='lchain-ddb', Key={'user_id': {'S': user}, 'session_id': {'S': session}})
    messages = response['Item']['History']['L']
    with st.expander("Chat History"):
        for message in reversed(messages):
            message_type = message['M']['type']['S']
            content = message['M']['data']['M']['content']['S']
            if message_type == 'human':
                st.markdown(f"**User:**\n {content}")
                st.markdown("---")
            elif message_type == 'ai':
                st.markdown(f"**AI:**\n {content}")