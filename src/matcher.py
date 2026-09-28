from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocess import clean_text
from src.extractor import extract_attributes


# Load AI embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def semantic_similarity(text1, text2):

    text1 = clean_text(text1)
    text2 = clean_text(text2)

    embeddings = model.encode(
        [text1, text2]
    )

    score = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return float(score)


def technical_comparison(attr1, attr2):

    fields = [
        "material_type",
        "size",
        "pressure_class",
        "connection",
        "material_grade",
        "standard"
    ]

    matched = 0
    compared = 0
    conflicts = []

    for field in fields:

        value1 = attr1.get(field)
        value2 = attr2.get(field)

        # If neither material has this attribute,
        # don't count it.
        if value1 is None and value2 is None:
            continue

        compared += 1

        if value1 == value2 and value1 is not None:

            matched += 1

        elif value1 is not None and value2 is not None:

            conflicts.append(
                f"{field}: {value1} ≠ {value2}"
            )

    if compared == 0:
        technical_score = 0
    else:
        technical_score = matched / compared

    return technical_score, conflicts


def compare_materials(description1, description2):

    # Clean descriptions
    cleaned1 = clean_text(description1)
    cleaned2 = clean_text(description2)

    # Extract technical attributes
    attr1 = extract_attributes(cleaned1)
    attr2 = extract_attributes(cleaned2)

    # AI semantic similarity
    semantic_score = semantic_similarity(
        cleaned1,
        cleaned2
    )

    # Technical comparison
    technical_score, conflicts = technical_comparison(
        attr1,
        attr2
    )

    # ---------------------------------
    # FINAL SCORE
    # ---------------------------------

    final_score = (
        0.50 * semantic_score
        +
        0.50 * technical_score
    )

    # ---------------------------------
    # DECISION
    # ---------------------------------

    # Major technical conflict
    if conflicts:

        decision = "TECHNICAL CONFLICT"

    elif final_score >= 0.85:

        decision = "LIKELY SAME MATERIAL"

    elif final_score >= 0.65:

        decision = "REVIEW REQUIRED"

    else:

        decision = "LIKELY DIFFERENT"

    return {
        "semantic_score": semantic_score,
        "technical_score": technical_score,
        "final_score": final_score,
        "decision": decision,
        "attributes_1": attr1,
        "attributes_2": attr2,
        "conflicts": conflicts
    }