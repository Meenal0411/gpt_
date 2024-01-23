import streamlit as st
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image

# Set the option to suppress the warning related to caching
# st.set_option('deprecation.showPyplotGlobalUse', False)

# Load GPT-2 model and tokenizer
gpt2_model_name = "gpt2"
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)

# Function to generate GPT-2 response with caching
@st.cache_data(show_spinner=False)
def generate_gpt2_response(prompt_input):
    input_ids = gpt2_tokenizer.encode(prompt_input, return_tensors="pt")
    output = gpt2_model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95,
                                 temperature=0.7)
    return gpt2_tokenizer.decode(output[0], skip_special_tokens=True)

def main():
    # App title
    st.set_page_config(page_title="🦙💬 Chatbot")

    # Display chat interface
    st.sidebar.title('🦙💬 Chatbot Menu')
    st.sidebar.write('Choose a chatbot to interact with.')

    option = st.sidebar.selectbox("Select a Chatbot", ["GPT-2", "About Us"])

    # Display or clear chat messages
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    if st.sidebar.button('Clear Chat History', on_click=clear_chat_history):
        return

    # User-provided prompt
    if option == "GPT-2":
        st.markdown("## Welcome to the GPT-2 Chatbot")

        user_input = st.text_input("You:", "")

        if st.button("Send"):
            if not user_input:
                st.warning("Please enter input.")
            else:
                user_input = user_input.strip()

                if not any(c.isalpha() for c in user_input):
                    st.warning("Invalid input. Please enter a meaningful input.")
                else:
                    response = generate_gpt2_response(user_input)
                    st.text("Bot: {}".format(response))

    elif option == "About Us":
        st.markdown("## About Us")
        st.write("This chatbot is powered by Hugging Face's transformers library. It uses a GPT-2 language model for general conversation.")
        st.write("Feel free to interact with the chatbot on the 'Home' page!")

        # Add an image in the About Us page (modify the path to your image)
        #about_us_image = Image.open("C:/Users/Sachin/OneDrive/Documents/New/chatbot.png")
        #st.image(about_us_image, caption="Another Image", use_column_width=True)

if __name__ == "__main__":
    main()
