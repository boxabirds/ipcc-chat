import pandas as pd
import h5py
import sys
from sqlalchemy import create_engine, inspect
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

def read_hdf5_to_dataframe(filepath, key='data'):
    """
    Reads a pandas DataFrame stored in an HDF5 file.

    :param filepath: str, the path to the HDF5 file.
    :param key: str, the key or the name of the group within the HDF5 file where the DataFrame is stored.
    :return: DataFrame containing the data from the HDF5 file.
    """
    df = pd.read_hdf(filepath, key)
    return df

def main():
    if len(sys.argv) < 3:
        print("Usage: python app.py <file_path> <query>")
        sys.exit(1)

    file_path = sys.argv[1]
    query = sys.argv[2]

    engine = create_engine("sqlite:///temps.db")
    inspector = inspect(engine)

    # Check if the 'temps' table already exists
    if 'temps' not in inspector.get_table_names():
        print(f"=== loading {file_path} ===")
        df = read_hdf5_to_dataframe(file_path)
        print(f"=== creating dataframe agent ===")
        df.to_sql("temps", engine, index=False, if_exists='replace')
    else:
        print("Data already loaded into database.")

    # Execute query using natural language
    db = SQLDatabase(engine=engine)
    print(db.dialect)
    print(db.get_usable_table_names())
    print(db.run("SELECT count(*) FROM model;"))

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
    agent_executor.invoke({"input": query})

if __name__ == "__main__":
    main()
