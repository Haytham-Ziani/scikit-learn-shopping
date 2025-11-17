import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    
    # Dictionary to map month abbreviations to integers
    months_to_int = {
        "Jan": 0,
        "Feb": 1,
        "Mar": 2,
        "Apr": 3,
        "May": 4,
        "June": 5,
        "Jul": 6,
        "Aug": 7,
        "Sep": 8,
        "Oct": 9,
        "Nov": 10,
        "Dec": 11
    }

    # Dictionary to map "TRUE" and "FALSE" strings to 1 and 0
    boolean_to_int = {
        "FALSE": 0,
        "TRUE": 1
    }

    evidence = []
    labels = []

    # Open CSV file and read its content
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader) # skip header
        for row in csv_reader:
            evidence_row = []
            # Convert each column to the correct data type
            evidence_row.append(int(row[0]))  # Administrative
            evidence_row.append(float(row[1]))  # Administrative_Duration 
            evidence_row.append(int(row[2]))  # Informational 
            evidence_row.append(float(row[3]))  # Informational_Duration 
            evidence_row.append(int(row[4]))  # ProductRelated 
            evidence_row.append(float(row[5]))  # ProductRelated_Duration 
            evidence_row.append(float(row[6]))  # BounceRates
            evidence_row.append(float(row[7]))  # ExitRates 
            evidence_row.append(float(row[8]))  # PageValues 
            evidence_row.append(float(row[9]))  # SpecialDay 
            evidence_row.append(months_to_int[row[10]])  # Month 
            evidence_row.append(int(row[11]))  # OperatingSystems 
            evidence_row.append(int(row[12]))  # Browser
            evidence_row.append(int(row[13]))  # Region 
            evidence_row.append(int(row[14]))  # TrafficType 
            
            visitor_type = 1 if row[15] == "Returning_Visitor" else 0
            evidence_row.append(visitor_type)  # VisitorType
            
            evidence_row.append(boolean_to_int[row[16]])  # Weekend
            
            evidence.append(evidence_row)
            
            labels.append(boolean_to_int[row[17]])

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    true_positives = 0
    true_negatives = 0
    total_positives = 0
    total_negatives = 0

    # Iterate over all labels and predictions
    for label, prediction in zip(labels, predictions):
        if label == 1:
            total_positives += 1
            if prediction == 1:
                true_positives += 1
        elif label == 0:
            total_negatives += 1
            if prediction == 0:
                true_negatives += 1

    sensitivity = true_positives / total_positives
    specificity = true_negatives / total_negatives

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
