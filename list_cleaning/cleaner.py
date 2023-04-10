def process_row(row):
    elements = row.split(" | ")

    # Check if there are enough elements to remove
    if len(elements) >= 4:
        del elements[2]
        del elements[-1]

    new_row = " | ".join(elements)
    return new_row

def contains_delisted(row):
    return 'delisted' in row.lower()

with open("/home/dan/langchain_projects/gptGoogle/documents/input_cal.txt", "r") as input_file:
    input_list = input_file.readlines()

output_list = [process_row(row.strip()) for row in input_list if not contains_delisted(row.strip())]

with open("cal.txt", "w") as output_file:
    for row in output_list:
        output_file.write(row + "\n")
