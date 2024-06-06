
# Talk to a Database  

This is an end to end LLM project based on OpenAI and Langchain. We are building a system that can talk to MsSQL database. 
User asks questions in a natural language and the system generates answers by converting those questions to an SQL query and
then executing that query on MySQL database. 
Paris data in MySQL database. A project manager 
will may ask questions such as,
- How many accounts are there in the database?
- List all group having platform name 'Linux Server?
The system is intelligent enough to generate accurate queries for given question and execute them on MySQL database



## Project Highlights

- Paris is a Previledge account access db 
- Data stored in a MySQL database
- We will build an LLM based question and answer system that will use following,
  - OpenAI
  - Hugging face embeddings
  - Streamlit for UI
  - Langchain framework
  - Chromadb as a vector store
  - Few shot learning
- In the UI, store manager will ask questions in a natural language and it will produce the answers


## Installation

1.Clone this repository to your local machine using:

```bash
  git clone https://github.com/Vandajai/Langchain_GenAI.git
```
2.Navigate to the project directory:

```bash
  cd Generativeai
```
3. Install the required dependencies using pip:

```bash
  pip install -r requirements.txt
```
4.Acquire an api key through makersuite.google.com and put it in .env file

```bash
  GOOGLE_API_KEY="your_api_key_here"
```
5. For database setup, run database/db_creation in MySQL workbench

## Usage

1. Run the Streamlit app by executing:
```bash
streamlit run main.py

```

2.The web app will open in your browser where you can ask questions

## Sample Questions
  - List all accounts?
  - Provide group name for account name root?
  - List all group having platform name 'Linux Server?

  
## Project Structure

- main.py: The main Streamlit application script.
- langchain_helper.py: This has all the langchain code
- requirements.txt: A list of required Python packages for the project.
- few_shots.py: Contains few shot prompts
- .env: Configuration file for storing your Google API key.