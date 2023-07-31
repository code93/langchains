from dotenv import load_dotenv
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import AgentType, create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools.python.tool import PythonREPLTool

load_dotenv()

# NEEDS GPT_$ might throw errors with gpt -3.5
def main():
    print("start...")
    python_agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # python_agent_executor.run(
    #     """generate and save in current working directory 15 QR codes that point to www.udemy.com/course/langchain, you have qr code package installed already."""
    # )

    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        path="episode_info.csv",
        verbose=True,
        AgentType=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

 #   csv_agent.run("how many columns are there in file episdoe_info.csv")
   # csv_agent.run("In file episode_info, which writer wrote the least episodes? How many episodes did he write?")
    csv_agent.run("Which season has most episodes")

if __name__ == "__main__":
    main()
