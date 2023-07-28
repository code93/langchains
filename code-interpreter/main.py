from dotenv import load_dotenv
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools.python.tool import PythonREPLTool

load_dotenv()


def main():
    print("start...")
    python_agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    python_agent_executor.run("""generate and save in current working directory 15 QR codes that point to www.udemy.com/course/langchain, you have qr code package installed already.""")

if __name__ == "__main__":
    main()
