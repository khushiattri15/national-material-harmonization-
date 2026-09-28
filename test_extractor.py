from src.preprocess import clean_text
from src.extractor import extract_attributes


description = 'BALL VLV 2 IN CL300 RF SS304'

cleaned = clean_text(description)

print("Original:")
print(description)

print("\nCleaned:")
print(cleaned)

print("\nExtracted Attributes:")

attributes = extract_attributes(cleaned)

for key, value in attributes.items():
    print(f"{key}: {value}")