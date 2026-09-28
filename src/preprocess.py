import re


# Common industrial abbreviations
ABBREVIATIONS = {
    "VLV": "VALVE",
    "VAL": "VALVE",
    "CL": "CLASS",
    "RF": "RAISED FACE",
    "FF": "FLAT FACE",
    "BW": "BUTT WELD",
    "SW": "SOCKET WELD",
    "CS": "CARBON STEEL",
    "SS": "STAINLESS STEEL",
    "CI": "CAST IRON",
    "DI": "DUCTILE IRON",
    "SCH": "SCHEDULE",
    "THK": "THICKNESS",
    "FLG": "FLANGE",
    "PMP": "PUMP",
}


def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.upper()

    # Replace special symbols
    text = text.replace('"', ' INCH ')
    text = text.replace("'", " FT ")

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Expand abbreviations
    words = text.split()

    cleaned_words = []

    for word in words:
        if word in ABBREVIATIONS:
            cleaned_words.append(ABBREVIATIONS[word])
        else:
            cleaned_words.append(word)

    text = " ".join(cleaned_words)

    return text