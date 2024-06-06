
#from langchain_helper import get_few_shot_db_chain
import os
#Import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain_core.pydantic_v1 import BaseModel, Field
import openai
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from typing import List
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent, ZeroShotAgent
from langchain.prompts.chat import ChatPromptTemplate
import langchain.globals
langchain.globals.set_verbose(True)  # Or False, depending on your needs
verbose = langchain.globals.get_verbose()
import pandas as pd
#from sql_execution import execute_sql_query
from langchain.prompts import load_prompt
from pathlib import Path
from langchain.chains import LLMChain
from sqlalchemy import create_engine
from langchain_experimental.sql import SQLDatabaseChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from few_shots import few_shots
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate,PromptTemplate
from langchain.chains import create_sql_query_chain



def main():
    load_dotenv()

    openai.api_type = os.getenv('OPENAI_API_TYPE')
    openai.api_version = os.getenv('OPENAI_API_VERSION')
    openai.azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    openai.api_key = os.getenv("OPENAI_API_KEY")

    username = "dbWSS"
    password = "LW)ppknteP'"
    host = "used.database.windows.net"
    database = "USED"
    driver = "ODBC Driver 17 for SQL Server"

   

    connection_string = (
        r'mssql+pyodbc://dbWSS:*JlaIl'
        r'@useq.database.windows.net/USEQ'
        r'?driver=ODBC+Driver+17+for+SQL+Server'

    )

    engine = create_engine(connection_string)
    db = SQLDatabase.from_uri(connection_string, schema="dbo",sample_rows_in_table_info=1,include_tables=['account','group','membership'])

    llm = AzureChatOpenAI(
        deployment_name="nlp-gpt4",
        temperature=0.2,
        max_tokens=900,
        azure_endpoint=openai.azure_endpoint
    )

    current_dir = Path(__file__)
    root_dir = [p for p in current_dir.parents if p.parts[-1]=='Paris'][0]
    print(root_dir)

    st.title("Paris OpenAI Search to Retrieve SQL Data")
    user_input = st.text_input("Enter your query")
    submit_button = st.button("Generate Answer")  # add a button
    tab_title=["Result","Query"]
    tabs = st.tabs(tab_title)

    #prompt_template = load_prompt(f"{root_dir}/few_shots.py")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    to_vectorize = [str(e) for e in few_shots]
    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
    )
    

    example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}\nSQLQuery:"),
        ("ai", "{query}"),
    ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
    input_variables=["input"],
)
    table_schema = db.get_table_info()
    final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a SQL server expert, who can execute and query SQL database to find answers based on user's question about tables available in the database and take only 3 tables for answering i.e account, membership and group table and try to join them together in order to get the result."\
         "Sometimes answers you can get from one table in that case you don't need to combine all 3 tables . First understand if requirements can be satisfying by using one table only if needed join all 3 tables [dbo].[account], [dbo].[membership] and [dbo].[group] if not getting answer from one table but do not go beyond 3 tables in the database." \
          "You should consider m_group_id column from [dbo].[membership] when joining with the [dbo].[group] table on id column and consider account_id when joining with [dbo].[account]"\
          "Here is the table schema: " + table_schema + \
          "Given an input question, create a syntactically correct  query to run, then look at the results of the query and return the answer in tabular format."\
          "Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 10 results."\
          "You can order the results by a relevant column to return the most interesting examples in the database."\
          "Never query for all the columns from a specific table, only ask for the relevant columns given the question."\
          "You have access to tools for interacting with the database."\
          "Only use the given tools. Only use the information returned by the tools to construct your final answer."\
          "You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again."\
          "DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database."\
          "If the question does not seem related to the database, just return 'I don't know' as the answer."\
          "Here are some examples of user inputs and their corresponding SQL queries" \
          "After joining all the tables if not getting any relevant answer pls return 'No data available' as the answer. Don't show any list or list of user names in this case"\
          "You don't want to see the entire data present in the table in the action input but wants to see final result.Also limit the number of rows to top 10 only when generating SQL queries .consider display name to user name in the account table."),
        ("system", "Example 1: If a user asks 'What are the account details for user XYZ?', you should generate a SQL query like 'SELECT Top 10 acct_name FROM account WHERE display_name = 'XYZ'"),
        ("system", "Example 2: If a user asks 'What groups is user XYZ a member of?', you should generate a SQL query like 'SELECT Top 10 [group].name FROM [group] JOIN membership ON [group].id = membership.group_id JOIN account ON membership.account_id = account.id WHERE account.display_name  = 'XYZ' LIMIT 10'.\n\nHere is the relevant table info: {}\n\nBelow are a number of examples of questions and their corresponding SQL queries.".format('[dbo].[group]')),

        few_shot_prompt,
        ("human", "{input}"),
    ]
)
 
    #Sql_generation_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=final_prompt)
    #Sql_generation_chain =  create_sql_query_chain(llm, db, final_prompt)
    Sql_generation_chain = LLMChain(llm=llm, prompt=final_prompt, verbose=True)

    if user_input:
        sql_query_dict = Sql_generation_chain(user_input)
        sql_query = sql_query_dict['text']  # extract the SQL query string
        result = pd.read_sql_query(sql_query, engine)

        with tabs[0]:
            st.write(result)
        with tabs[1]:
            st.write(sql_query)


if __name__ == "__main__":
    main()


