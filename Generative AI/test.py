import os
import openai
import streamlit as st
from langchain_helper import get_few_shot_db_chain
from sqlalchemy import create_engine
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
import pandas as pd
from langchain.chains import LLMChain

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

    
    st.title("Paris OpenAI Search to Retrieve SQL Data")
    user_input = st.text_input("Enter your query", key="unique_key")
    submit_button = st.button("Generate Answer")  # add a button    
    #tab_title=["Result","Query"]
    #tabs = st.tabs(tab_title)

    Sql_generation_chain = LLMChain(llm=llm, prompt=get_few_shot_db_chain(), verbose=True)
    
    if submit_button and user_input:
        sql_query_dict = Sql_generation_chain(user_input)
        sql_query = sql_query_dict['text']  # extract the SQL query string
        result = pd.read_sql_query(sql_query, engine)

        st.header("Result")
        st.write(result)

        st.sidebar.header("Query")
        st.sidebar.write(sql_query)


if __name__ == "__main__":
    main()



