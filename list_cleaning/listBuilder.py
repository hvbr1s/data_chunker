input_file = "cal.txt"
output_file = "cal.html"

# Read the input file and split its contents into a list
with open(input_file, "r") as f:
    lines = f.read().splitlines()

# Generate the HTML file with the list
with open(output_file, "w") as f:
    f.write("<ul>\n")
    for line in lines:
        f.write(f"<li>{line}</li>\n")
    f.write("</ul>")
