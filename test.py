import csv

months_to_int = {
    "Jan" : 0,
    "Feb" : 1,
    "Mar" : 2,
    "Apr" : 3,
    "May" : 4,
    "June" : 5,
    "Jul" : 6,
    "Aug" : 7,
    "Sep" : 8,
    "Oct" : 9,
    "Nov" : 10,
    "Dec" : 11
}

bool_dict = {
    "FALSE" : 0,
    "TRUE" : 1
}

returning_visitors = {
    "Returning_Visitor" : 1,
    "New_Visitor" : 0
}


with open("shopping_test.csv", 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)

    with open("shopping_2.csv", 'w') as new_file:
        csv_writer = csv.writer(new_file)
        for row in csv_reader:
            row[10] = months_to_int[row[10]]
            row[15] = returning_visitors.get(row[15], 0)
            row[16] = bool_dict[row[16]]
            row[17] = bool_dict[row[17]]

            csv_writer.writerow(row)