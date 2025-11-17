# OpenAI API is blocked
# import os
# from openai import OpenAI

# # Create an OpenAI client
# client = OpenAI(api_key=os.environ["OPENAI_API_TOKEN"])

# print(f"OpenAI client created successfully: {client}")


# Use boto3 to create Bedrock client
import boto3

# Create a session with the profile
session = boto3.Session(profile_name="488608459208_BedrockPilotUsers")

# Create the client using the session
# bedrock = session.client(service_name="bedrock", region_name="us-east-1")
# print(bedrock.list_foundation_models())

bedrock_runtime = session.client(
    service_name="bedrock-runtime", region_name="us-east-1"
)
