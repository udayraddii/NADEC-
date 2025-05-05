import os
import sys
import re
import json
import traceback
import torch
import urllib
from dotenv import load_dotenv

import streamlit as st
import pandas as pd
import pyodbc
from sqlalchemy import create_engine, MetaData
from sqlalchemy.engine import URL
from streamlit_mic_recorder import mic_recorder
# from groq import Groq

# ---------------------------
# External LLM & LangChain Imports
# ---------------------------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS
from deep_translator import GoogleTranslator 
import langdetect
from groq import Groq

# Load environment variables early.
load_dotenv()

# Initialize Groq client for audio transcription
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Function to transcribe audio
def transcribe_audio(filename):
    with open(filename, "rb") as file:
        transcription = groq_client.audio.transcriptions.create(
            file=(filename, file.read()),
            model="whisper-large-v3",
            prompt="Specify context or spelling",
            response_format="json",
            language=None,
            temperature=0.0
        )
    detected_lang = detect_language(transcription.text)  # Detect language separately
    return transcription.text, detected_lang
 

# Convert text to speech
def text_to_audio(text,lang=None):
    if lang is None:
        if lang is None:
            lang="en"
            
    tts = gTTS(text=text, lang='en', slow=False)
    mp3_file = "temp_audio.mp3"
    tts.save(mp3_file)
    return mp3_file

def detect_language(text):
    """Detects the language of the input text."""
    try:
        return langdetect.detect(text)
    except:
        return "en"  # Default to English if detection fails

def translate_text(text, src_lang, dest_lang):
    """Translates text from src_lang to dest_lang."""
    if src_lang == dest_lang:
        return text  # No translation needed
    return GoogleTranslator(source=src_lang, target=dest_lang).translate(text)


# =================================================================================================================
# AGENT 1: CLASSIFIER
# ================================================================================================================
class Agent1Classifier:
    """
    Agent 1: Determines if the user's query calls for a visualization ("viz")
    or a textual explanation ("text") using a heuristic and an instruction-tuned model.
    """
    def __init__(self, model_name: str = "google/flan-t5-base", device: str = "cpu"):
        # Heuristic keywords that clearly indicate visualization.
        self.visual_keywords = [
           "bar chart", "line chart", "scatter plot", "pie chart", "heatmap","box plot",
            "violin plot", "histogram", "area chart", "bubble chart", "radar chart", 
            "donut chart", "stacked bar chart", "stacked area chart", "stacked line chart",
            "stacked scatter plot", "stacked histogram", "stacked donut chart", "stacked pie chart",
            "stacked bubble chart", "stacked violin plot", "stacked box plot", "stacked radar chart",
            "grouped bar chart", "grouped area chart", "grouped line chart", "grouped scatter plot",
            "grouped histogram", "grouped donut chart", "grouped pie chart", "grouped bubble chart",
            "grouped violin plot", "grouped box plot", "grouped radar chart", "grouped heatmap",
            "grouped stacked bar chart", "grouped stacked area chart", "grouped stacked line chart",
            "grouped stacked scatter plot", "grouped stacked histogram", "grouped stacked donut chart",
            "graph", "visualize", "visualization", "diagram", "chart", "plot", "graph", "visualisation", 
            "trend"

        ]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipeline = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1
        )

    def classify_query(self, user_query: str) -> str:
        query_lower = user_query.lower()
        # Heuristic: if any explicit visualization keyword is found, return "viz".
        for kw in self.visual_keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", query_lower):
                return "viz"
        # Otherwise, use the instruction-tuned model.
        prompt = (
            "You are an expert classifier. Based on the following query, decide whether "
            "the user wants a data visualization or a textual explanation. "
            "Answer with exactly one word: "
            "type 'viz' if the user is asking for a chart, graph, plot, or any visual representation; "
            "otherwise, answer 'text'.\n\n"
            f"User query: \"{user_query}\" \nAnswer:"
        )
        output = self.pipeline(prompt, max_new_tokens=5)[0]["generated_text"]
        answer = output.strip().lower().split()[0]
        return answer if answer in ["viz", "text"] else "text"


# ===========================
# AGENT 2: TEXT INSIGHTS
# ===========================
class AgentText:
    """
    Agent that converts a natural language query into SQL, executes it,
    and returns a descriptive answer with the query results.
    """
    def __init__(self):
        # Pull DB credentials from environment.
        self.SQL_SERVER = os.getenv("SQL_SERVER")
        self.SQL_DATABASE = os.getenv("SQL_DATABASE")
        self.SQL_USERNAME = os.getenv("SQL_USERNAME")
        self.SQL_PASSWORD = os.getenv("SQL_PASSWORD")
        self.GROQ_API_KEY = "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"

        # Create SQLAlchemy Connection URL.
        self.conn_url = URL.create(
            "mssql+pyodbc",
            username=self.SQL_USERNAME,
            password=self.SQL_PASSWORD,
            host=self.SQL_SERVER,
            port=1433,
            database=self.SQL_DATABASE,
            query={
                "driver": "ODBC Driver 18 for SQL Server",
                "Encrypt": "yes",
                "TrustServerCertificate": "no"
            },
        )

        # Test the pyodbc connection.
        try:
            test_conn = pyodbc.connect(
                f"DRIVER={{ODBC Driver 18 for SQL Server}};"
                f"SERVER={self.SQL_SERVER};"
                f"DATABASE={self.SQL_DATABASE};"
                f"UID={self.SQL_USERNAME};"
                f"PWD={self.SQL_PASSWORD};"
                "Encrypt=yes;"
                "TrustServerCertificate=no;"
                "Connection Timeout=30;"
            )
            test_conn.close()
            st.info("✅ Database connection successful (Text Agent)!")
        except Exception as e:
            st.error("❌ Error testing database connection (Text Agent): " + str(e))
            raise

        # Create SQLAlchemy Engine and reflect metadata.
        self.db_engine = create_engine(self.conn_url, fast_executemany=True)
        metadata = MetaData()
        metadata.reflect(bind=self.db_engine)
        st.write(f"Tables in database: {list(metadata.tables.keys())}")

        # Initialize SQLDatabase object.
        self.db = SQLDatabase(self.db_engine, view_support=True, schema="dbo")

        # Initialize Groq LLM.
        self.llm = ChatGroq(
            api_key=self.GROQ_API_KEY,
            model="llama3-70b-8192",
            temperature=0.7,
            max_tokens=8192,
            timeout=None,
            max_retries=2,
        )

        # Define a system message that tells the agent to produce a descriptive final answer.
        system_message = (
            "You are a helpful agent that can run SQL queries on a database to retrieve data. "
            "Return the final answer in a descriptive, natural language format. "
            "If the user references columns or tables that don't exist, clarify or correct them."
        )

        # Create the SQL Database Toolkit and the SQL Agent.
        # We rely on the default ReAct prompt so it will:
        #  1) Generate the SQL query
        #  2) Execute it via the SQL tool
        #  3) Return the final answer in plain text
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.agent_executor = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            verbose=False,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,  # so it retries if there's a parsing error
            system_message=system_message
        )

    def query(self, user_query: str) -> str:
        """
        Invoke the agent with a natural language query and return the final answer
        (which includes the data results) in a descriptive manner.
        """
        try:
            # Using .run() so the final output is a single string with the descriptive answer.
            # (Alternatively, you could use .invoke() which returns a dict.)
            response = self.agent_executor.run(user_query)
            return response
        except Exception as e:
            err = f"❌ Error processing query in AgentText: {e}\n{traceback.format_exc()}"
            return err



# ===========================
# AGENT 3: VISUALIZATION
# ===========================
# Function to establish a database connection
def get_db_engine():
    SQL_SERVER = os.getenv("SQL_SERVER")
    SQL_DATABASE = os.getenv("SQL_DATABASE")
    SQL_USERNAME = os.getenv("SQL_USERNAME")
    SQL_PASSWORD = os.getenv("SQL_PASSWORD")
    connection_string = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={SQL_SERVER};"
        f"DATABASE={SQL_DATABASE};"
        f"UID={SQL_USERNAME};"
        f"PWD={SQL_PASSWORD};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )
    params = urllib.parse.quote_plus(connection_string)
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
    return engine

# Function to rewrite user query into a structured visualization request
def rewrite_query(query):
    query = query.lower()
    if "show" in query and "as" in query:
        query = re.sub(r"show (.*) as (.*)", r"Visualize \1 using a \2.", query)
    if "by" in query:
        query = re.sub(r"(\b.*\b) by (\b.*\b)", r"Group \1 by \2.", query)
    if not any(chart_type in query for chart_type in ["bar chart", "line chart", "scatter plot"]):
        query += " Use the most suitable chart type."
    return query.capitalize()

# Function to generate visualization code using LLM
def generate_visualization_code(prompt, df):
    columns = df.columns.tolist()
    data_types = df.dtypes.apply(lambda x: str(x)).to_dict()
    rewritten_prompt = rewrite_query(prompt)
    modified_prompt = (
        f"Generate an interactive Plotly visualization for the given DataFrame. "
        f"The columns are {columns} with the following data types: {data_types}. "
        f"Ensure that the code is syntactically correct, automatically resolve any potential errors."
        f"Ensure it is interactive with hover tooltips, proper axis labels, a legend, and titles for better aesthetics. "
        f"Assign the visualization to the variable 'plot_output' so it can be rendered directly. "
        f"User requested: {rewritten_prompt}"
    )
    
    viz_api_key = os.getenv("GROQ_API_KEY", "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")
    chat_groq = ChatGroq(model="llama3-70b-8192", api_key=viz_api_key)
    response = chat_groq.invoke(modified_prompt)
    return extract_code_from_response(response)

# Function to extract Python code from LLM response
def extract_code_from_response(response):
    content = response.content
    if "```" in content:
        code_parts = content.split("```")
        for part in code_parts:
            if "import" in part:  # Likely the code snippet.
                cleaned_code = part.strip().replace("python", "").replace("Python", "").strip()
                return cleaned_code
    return content

# Function to regenerate faulty code based on error message
def regenerate_code_with_fix(code, error_msg):
    prompt = (
        f"The following Python code for a Plotly visualization contains an error:\n\n"
        f"```python\n{code}\n```\n\n"
        f"Error message: {error_msg}\n\n"
        "Please correct the code and ensure it runs successfully without errors. "
        "Make sure the variable 'plot_output' contains the visualization."
    )
    
    viz_api_key = os.getenv("GROQ_API_KEY", "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")
    chat_groq = ChatGroq(model="llama3-70b-8192", api_key=viz_api_key)
    response = chat_groq.invoke(prompt)
    
    return extract_code_from_response(response)

# Function to execute code and retry if errors occur
def execute_code_and_plot(code, df, max_retries=3):
    local_scope = {"df": df, "plot_output": None}
    
    for attempt in range(max_retries):
        try:
            exec(code, {}, local_scope)
            plot_output = local_scope.get("plot_output", None)
            
            if plot_output:
                plot_output.update_layout(height=400, width=600)
                # description = generate_chart_description(code)
                # st.write(f"**Chart Summary:** {description}")
                # aud_file = text_to_audio(description)
                # st.audio(aud_file, format="audio/mp3")
                return plot_output
            else:
                st.warning("Code executed, but no plot was generated. Regenerating...")
        
        except Exception as e:
            error_msg = str(e)
            st.error(f"Error in execution: {error_msg}")

            if attempt < max_retries - 1:
                st.info("Regenerating visualization code...")
                code = regenerate_code_with_fix(code, error_msg)
            else:
                st.error("Failed to generate a valid visualization after multiple attempts.")
                return None
            
    return None

# Function to generate a chart description using LLM
def generate_chart_description(code):
    
    prompt = (
        "Analyze the following Python code that generates a Plotly visualization:\n\n"
        f"```python\n{code}\n```\n\n"
        "Based on the code (which shows the chart type, axes, and any grouping) and this sample of data:\n"
        "1. Provide a concise paragraph (2-4 sentences) describing what the chart shows.\n"
        "2. Mention any major insights or numeric patterns, such as which categories or columns stand out, significant differences, or interesting trends.\n"
        "3. Reference approximate numbers or ranges if you can deduce them from the sample.\n"
        "Make sure the description is clear, uses natural language, and does not simply restate the code."
    )
    
    viz_api_key = os.getenv("GROQ_API_KEY", "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")
    chat_groq = ChatGroq(model="llama3-70b-8192", api_key=viz_api_key)
    response = chat_groq.invoke(prompt)
    
    return response.content.strip()

# Function to load data from a SQL table
@st.cache_data(show_spinner=False)
def load_table_data(table_name: str) -> pd.DataFrame:
    engine = get_db_engine()
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, engine)
    engine.dispose()
    return df

# Function to get available table names
@st.cache_data(show_spinner=False)
def get_table_names() -> list:
    engine = get_db_engine()
    query_tables = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
    tables_df = pd.read_sql(query_tables, engine)
    engine.dispose()
    return tables_df["TABLE_NAME"].tolist()

# Function to choose the most relevant table for visualization
def choose_table(query, table_names):
    prompt = (
        f"Available tables: {', '.join(table_names)}. "
        f"Based on the user's query: '{query}', which table is most relevant for generating a visualization? "
        "Respond with only the table name."
    )
    
    viz_api_key = os.getenv("GROQ_API_KEY", "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")
    chat_groq = ChatGroq(model="llama3-70b-8192", api_key=viz_api_key)
    response = chat_groq.invoke(prompt)
    
    chosen_table = response.content.strip()
    if chosen_table not in table_names:
        st.warning(f"LLM suggested table '{chosen_table}' which is not in the available list. Defaulting to {table_names[0]}.")
        chosen_table = table_names[0]
    
    return chosen_table



# ===========================
# STREAMLIT MAIN APPLICATION
# ===========================


def main():
    # Configure the page layout and title.
    st.set_page_config(page_title="Analytics-Bot", layout="wide")
    
    # Sidebar: Instructions and guidelines.
    st.sidebar.header("How to Use")
    st.sidebar.markdown(
        """
        **Steps:**
        - **Enter your query:** Use the input box below.
        - **Automatic Detection:** The system determines if your query requires a visualization or textual insights.
        - **Visualization Tips:** For visual results, include keywords like *chart*, *graph*, *plot*, etc.
        - **Textual Insights:** Ask natural language questions for detailed explanations.
        """
    )
    st.sidebar.info("Example queries: 'Show sales by region as a bar chart' or 'What are total sales last quarter?'")

    # Main Title and Description.
    st.title("AI-Analytics")
    st.markdown(
        "Welcome to the Multi-Agent SQL Assistant. This tool leverages multiple agents to analyze your query and either generate a data visualization or provide textual insights directly from your SQL database. Simply type your query below and hit Enter."
    )

    # User Input Options: Voice or Text
    input_mode = st.radio("Choose Input Mode:", ["Voice Input", "Text Input"], horizontal=True)

    user_query = ""

    # Voice Input Processing
    if input_mode == "Voice Input":
        audio = mic_recorder(key="recorder", format="wav")
        if audio is not None:
            with st.spinner("Transcribing audio..."):
                filename = "temp_audio.wav"
                with open(filename, "wb") as f:
                    f.write(audio['bytes'])
                try:
                    user_query,detected_lang= transcribe_audio(filename)
                    st.success(f"Detected Language: **{detected_lang.upper()}**")
                    st.success(user_query)
                except Exception as e:
                    st.error(f"Error transcribing audio: {str(e)}")

    # Text Input Processing
    elif input_mode == "Text Input":
        user_query = st.text_input(
            "Enter your query:",
            placeholder="e.g., 'Show sales by region as a bar chart' or 'What are total sales last quarter?'",
            key="query_input"
        )

        if user_query:
            # Step 1: Detect Input Language
           
            detected_lang = detect_language(user_query)
            st.info(f"Detected Language: **{detected_lang.upper()}**")



    if user_query:

        # Step 2: Translate Query to English for Processing
        translated_query = translate_text(user_query, detected_lang, "en")
        st.info(f"Translated Query: **{translated_query}**")




        # Query Classification Section.
        with st.spinner("Classifying your query..."):
            classifier = Agent1Classifier()
            query_type = classifier.classify_query(translated_query)
        st.success(f"Detected Query Type: **{query_type.upper()}**")
        




        # Process as a textual query.
        if query_type == "text":
            st.subheader("Textual Query Result")
            try:
                agent_text = AgentText()
                result = agent_text.query(translated_query)




                # Step 3: Translate Response Back to Original Language
                final_response = translate_text(result, "en", detected_lang)

                st.markdown(final_response)
                st.markdown(result)

                aud_file = text_to_audio(final_response, lang=detected_lang)
                st.audio(aud_file, format='audio/mp3')

            except Exception as e:
                st.error(f"Error processing textual query: {e}")



        # Process as a visualization query.
        elif query_type == "viz":
            st.subheader("Visualization Query Result")
            try:
                table_names = get_table_names()
                if not table_names:
                    st.error("No tables found in the database.")
                    return

                with st.spinner("Determining the most appropriate table..."):
                    chosen_table = choose_table(translated_query, table_names)
                st.info(f"Selected Table: **{chosen_table}**")

                with st.spinner("Loading data from the selected table..."):
                    df = load_table_data(chosen_table)
                st.markdown("### Data Preview")
                st.dataframe(df.head())

                with st.spinner("Generating visualization code..."):
                    code = generate_visualization_code(translated_query, df)

                with st.spinner("Executing visualization code..."):
                    plot_output = execute_code_and_plot(code, df)

                if plot_output:
                    st.plotly_chart(plot_output, use_container_width=True)
            except Exception as e:
                st.error(f"Error processing visualization query: {e}")

if __name__ == "__main__":
    main()