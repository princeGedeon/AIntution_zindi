import pandas as pd
from tools.main import retrieve_contexts

# Read the input CSV file
df = pd.read_csv("./test_files/Test.csv")

# Output column names
outputs_names = [f"Output_{i}" for i in range(1, 6)]

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Retrieve the contexts for the given query
    tmp_contexts = retrieve_contexts(row['Query text'], num_contexts=5)

    # Ensure the retrieved contexts are of the expected length
    if len(tmp_contexts) == 5:
        # Update the DataFrame directly using df.at
        df.at[index, "Output_1"] = tmp_contexts[0]
        df.at[index, "Output_2"] = tmp_contexts[1]
        df.at[index, "Output_3"] = tmp_contexts[2]
        df.at[index, "Output_4"] = tmp_contexts[3]
        df.at[index, "Output_5"] = tmp_contexts[4]

# Display the updated DataFrame
print(df.head())

# Save the updated DataFrame to a new CSV file
df.to_csv("Test_generated.csv", sep=",", index=False)
