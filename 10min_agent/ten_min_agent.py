from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool

from dotenv import load_dotenv
load_dotenv()

import json
from typing import List
import random

# Write tools for the agent
@tool
def write_json(file_path:str, content:str) -> str:
    """Writes content to a JSON file at the specified file path."""
    try:
        data = json.loads(content)
        with open(file_path, 'w', encoding='utf-8') as fp:
            json.dump(data, fp, indent=2, ensure_ascii=False)
        print(f"JSON content written to {file_path}")
    except Exception as e:
        return f"Error writing JSON to {file_path}: {str(e)}"
    
@tool
def read_json(file_path:str) -> str:
    """Reads and returns content from a JSON file at the specified file path."""
    try:
        with open(file_path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error reading JSON from {file_path}: {str(e)}"
    
@tool
def generate_sample_users(first_names: List[str], last_names: List[str], min_age: int, max_age: int) -> str:
    """Generates a list of sample users in JSON format."""
    if not first_names:
        return f"Error: first_names list is empty."
    if not last_names:
        return f"Error: last_names list is empty."
    if min_age > max_age:
        return f"Error: min_age {min_age} is greater than max_age {max_age}."
    
    users = []
    count = len(first_names)

    # Generate user data
    for i in range(count):
        user = {
            "first_name": first_names[i],
            "last_name": last_names[i % len(last_names)],
            "email": f"{first_names[i].lower()}.{last_names[i % len(last_names)].lower()}@example.com",
            "age": random.randint(min_age, max_age)
        }
        users.append(user)
        
    return {"users": users, "count": len(users)}
    
# Create the agent
system_message = """You are a helpful assistant that helps users generate, read, and write JSON files containing user data. 
Use the provided tools to perform file operations and generate sample user data as needed. 
If the user requests generating random users, generate the specified number of users with random first names, random last names and ask for age range. 
Always ensure the JSON content is properly formatted.
After generating the user data, write it to a JSON file with the write_json tool."""
llm = ChatOpenAI(model="gpt-4", temperature=0)
agent = create_agent(model = llm, tools = [write_json, read_json, generate_sample_users], system_prompt=system_message)

# Function to run the agent
def run_agent(user_input: str, history: List[BaseMessage]) -> AIMessage:
    """Runs the agent with the given user input and returns the response."""
    try:
        response = agent.invoke(
        {"messages": history + [HumanMessage(content=user_input)]}, 
       config= {"recursion_limit": 50}
        )
        return response["messages"][-1] # Return the last message which is the agent's response
    except Exception as e:
        return AIMessage(content=f"Error during agent execution: {str(e)}")
    

# Main loop for user interaction
def main():
    history: List[BaseMessage] = []
    print("Welcome to the 10-Minute JSON Agent! Type 'exit' to quit.")
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            print("Exiting the agent. Goodbye!")
            break
        ai_message = run_agent(user_input, history)
        history += [HumanMessage(content=user_input), ai_message]

        print(f"Agent: {ai_message.content}")

# Call main when run as a script
if __name__ == "__main__":
    main()

    