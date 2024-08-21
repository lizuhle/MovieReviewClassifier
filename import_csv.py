import csv

fieldnames = ["ID", "TEXT"]
out_fields = ["ID", "LABEL"]

with open("test.csv") as file:
    reader = csv.DictReader(file, fieldnames=fieldnames)
    with open("test0.csv", "w") as output:
        writer = csv.DictWriter(output, fieldnames=out_fields)
        next(reader)
        writer.writeheader()
        for row in reader:
            writer.writerow({"ID": row["ID"], 'LABEL': 0})