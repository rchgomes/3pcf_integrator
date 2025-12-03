import itertools
import os

def generate_ini_files(template_content):
    ddE_variants = ["ddE","dEd", "Edd", "dEE", "EdE", "EEd", "EEE"]
    #ddE_variants = ["ddE"]
    range_variants = ["first1000","1000to2000", "2000to3000", "3000to4000", "4000to5000"]
    #range_variants = ["1000to2000"]

    for ddE_var, range_var in itertools.product(ddE_variants, range_variants):
        new_content = template_content.replace("ddE", ddE_var)
        new_content = new_content.replace("first1000", range_var)

        output_filename = f"generate_training_set_{ddE_var}_{range_var}.ini"
        with open(output_filename, "w") as f:
            f.write(new_content)
        print(f"Generated {output_filename}")

# Read the template content from the provided file
with open("generate_training_set_ddE_first1000.ini", "r") as f:
    template_ini_content = f.read()

generate_ini_files(template_ini_content)

