import os
from openai import OpenAI

# Create an OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_TOKEN"])

print(f"OpenAI client created successfully: {client}")
