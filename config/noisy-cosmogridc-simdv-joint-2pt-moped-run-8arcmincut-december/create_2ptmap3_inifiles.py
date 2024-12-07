# Python script to generate modified `.ini` files
original_file = "des-y3-shear-2pt-map3-NLA-all-001.ini"

# Read the original file content
with open(original_file, "r") as file:
    content = file.read()

# Loop to generate files from 002 to 050
for i in range(2, 51):
    # Generate the replacement number as a 3-digit string
    replacement = f"{i:03}"
    # Replace '001' in the content with the current replacement number
    modified_content = content.replace("001", replacement)
    # Create the new filename
    new_filename = f"des-y3-shear-2pt-map3-NLA-all-{replacement}.ini"
    # Write the modified content to the new file
    with open(new_filename, "w") as new_file:
        new_file.write(modified_content)

print("Files have been successfully generated!")