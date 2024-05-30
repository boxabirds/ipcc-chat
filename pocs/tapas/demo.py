import argparse
from transformers import pipeline
import xarray as xr

# Create an argument parser
parser = argparse.ArgumentParser(description="Natural Language Queries on .nc Files")
parser.add_argument("--file", help="Path to the .nc file")
args = parser.parse_args()

# Load the .nc file into an xarray Dataset
ds = xr.open_dataset(args.file)
print("=== dataset ===")
print(ds)
print("=== dataset as dict ===")
table_dict = ds.to_dict()
print(table_dict)

print("=== dimensions ===")
# Access the dimensions
for dim in ds.dims:
    print(f"Dimension: {dim}, Size: {ds.dims[dim]}")

print("=== Columns ===")
# Access the variables and their metadata
for var in ds.variables:
    print(f"\nVariable: {var}")
    print(f"Dimensions: {ds[var].dims}")
    print(f"Data Type: {ds[var].dtype}")
    print(f"Attributes:")
    for attr in ds[var].attrs:
        print(f"  {attr}: {ds[var].attrs[attr]}")

# Initialize a question-answering pipeline using a pre-trained model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Define a function to query the dataset using natural language
def query_dataset(question):
    # Convert the xarray Dataset to a string representation
    context = str(ds)
    
    # Pass the question and context to the question-answering pipeline
    result = qa_pipeline(question=question, context=context)
    
    return result["answer"]



# # Example usage
# question = "What is the average temperature in the dataset?"
# answer = query_dataset(question)

# print(f"question: {question}")
# print(f"answer: {answer}")

