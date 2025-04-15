from huggingface_hub import InferenceClient
from config import BASE_MODEL, MY_MODEL, HF_TOKEN
import os
class SchoolChatbot:
    """
    This class is extra scaffolding around a model. Modify this class to specify how the model recieves prompts and generates responses.

    Example usage:
        chatbot = SchoolChatbot()
        response = chatbot.get_response("What schools offer Spanish programs?")
    """

    def __init__(self):
        """
        Initialize the chatbot with a HF model ID
        """
        model_id = MY_MODEL if MY_MODEL else BASE_MODEL # define MY_MODEL in config.py if you create a new model in the HuggingFace Hub
        self.client = InferenceClient(model=model_id, token=HF_TOKEN)
        
        contexts = []
        
        # Read each text file in corpus directory
 
        corpus_dir = "corpus"
        for filename in os.listdir(corpus_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(corpus_dir, filename), 'r') as f:
                    contexts.append(f.read())
        self.contexts = contexts
    def join_context(self, prompt, context_string, prev_summary):
        """
        Generates a summarized context by combining new context with previous context, focused on relevance to the prompt.

        Args:
            prompt (str): The current user query or prompt to focus the summary around
            context_string (str): New context information to incorporate
            prev_summary (str): Previous context summary to build upon

        Returns:
            str: A summarized context combining relevant information from both sources, limited to 4000 characters

        The method uses the language model to:
        1. Extract information relevant to the prompt from both contexts
        2. Combine and summarize the information
        3. Return a coherent summary limited to 4000 characters
        """
        
        # Generate a summary using the model
        prompt = f'''Using at most 4000 characters, please create a new summaty which encapsulates all information relevant to
        the given PROMPT from CONTEXT or PREV_CONTEXT
        \n\nPROMPT:{prompt}\n\n
        \n\nCONTEXT:{context_string}\n\n
        \n\nPREV_CONTEXT:{prev_summary}\n\n
        Summary:'''
        summary = self.client.text_generation(
            prompt,
            max_new_tokens=100,
            temperature=0.3,
            repetition_penalty=1.2,
            return_full_text=False
        )
        return summary
    def format_prompt(self, user_input):
        """
        TODO: Implement this method to format the user's input into a proper prompt.
        
        This method should:
        1. Add any necessary system context or instructions
        2. Format the user's input appropriately
        3. Add any special tokens or formatting the model expects

        Args:
            user_input (str): The user's question about Boston schools

        Returns:
            str: A formatted prompt ready for the model
        
        Example prompt format:
            "You are a helpful assistant that specializes in Boston schools...
             User: {user_input}
             Assistant:"
        """
        summary = ""
                    
        for context in self.contexts:
            summary = self.join_context(user_input, context, summary)
            
        prompt = f"""You are a helpful assistant that specializes in Boston schools who aims to help families
        understand and navigate the public school website to understand which schools they can register for. The information
        you from the website you need to answer the User Question is provided below as Context.
             User Question: {user_input}
             Context: {summary}
        """
        return prompt

        
    def get_response(self, user_input):
        """
        TODO: Implement this method to generate responses to user questions.
        
        This method should:
        1. Use format_prompt() to prepare the input
        2. Generate a response using the model
        3. Clean up and return the response

        Args:
            user_input (str): The user's question about Boston schools

        Returns:
            str: The chatbot's response

        Implementation tips:
        - Use self.format_prompt() to format the user's input
        - Use self.client to generate responses
        """
        #format full prompt
        prompt = self.format_prompt(user_input)

        # Generate response using the model
        response = self.client.generate(prompt)
        
        # Clean and return the response
        # Remove any "Assistant:" prefix if present
        # response = response.replace("Assistant:", "").strip()
        
        return response