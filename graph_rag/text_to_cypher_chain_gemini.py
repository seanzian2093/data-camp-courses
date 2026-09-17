import os
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph

# Initialize the language model (using Google Gemini)

llm = ChatGoogleGenerativeAI(
    google_api_key=os.getenv("GEMINI_API_KEY"),
    model="gemini-3.6-flash",
    temperature=0
)

# Initialize the Neo4j graph
graph = Neo4jGraph()

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(""" 
		You are an expert Neo4j developer. Use the following database schema to write a Cypher statement to answer the user's question. Only generate the Cypher statement, no pre-amble. Do not return any Markdown.
	Schema: 
    {schema}
    
	Question: {question}""", 
	# Update the partial_variables dictionary
    partial_variables={"schema": graph.schema})
])

# Create the text-to-Cypher chain
text_to_cypher_chain = prompt | llm | StrOutputParser()
cypher = text_to_cypher_chain.invoke({"question": "Where does Jo Cornelissen work?"})
print(cypher)

