# Import necessary libraries
import os

# Read the original file
original_filename = "/Users/gchgomes/3pcf_integrator/config/noisy-cosmogridc-simdv-joint-2pt-moped-run-8arcmincut/des-y3-shear-2pt-map3-NLA-all-001.ini"
with open(original_filename, "r") as file:
    content = file.read()

# Generate 47 files
for i in range(4, 51):  # 004 to 050
    suffix = f"{i:03}"  # Ensure zero-padded numbering
    new_filename = f"/Users/gchgomes/3pcf_integrator/config/noisy-cosmogridc-simdv-joint-2pt-moped-run-8arcmincut/des-y3-shear-2pt-map3-NLA-all-{suffix}.ini"

    # Replace occurrences of "001" with the new suffix
    new_content = content.replace("001", suffix)

    # Write the new content to a new file
    with open(new_filename, "w") as new_file:
        new_file.write(new_content)

print("47 files successfully created.")
