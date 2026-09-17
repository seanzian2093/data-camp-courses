import os
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph

# Initialize the language model
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo",
    temperature=0
)
# Initialize the Neo4j graph
graph = Neo4jGraph()

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(""" 
		You are an expert Neo4j developer. 
        Use the following database schema to write a Cypher statement to answer the user's question. 
        Only generate the Cypher statement, no pre-amble. Do not return any Markdown.
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