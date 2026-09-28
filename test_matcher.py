from src.matcher import compare_materials


material_a = "BALL VLV 2 IN CL300 RF SS304"

material_b = "BALL VALVE DN50 CLASS 150 RAISED FACE SS304"


result = compare_materials(
    material_a,
    material_b
)


print("\nMATERIAL A:")
print(material_a)

print("\nMATERIAL B:")
print(material_b)

print("\n-------------------------")

print(
    "Semantic Score:",
    round(result["semantic_score"] * 100, 2),
    "%"
)

print(
    "Technical Score:",
    round(result["technical_score"] * 100, 2),
    "%"
)

print(
    "Final Score:",
    round(result["final_score"] * 100, 2),
    "%"
)

print(
    "Decision:",
    result["decision"]
)

print("\nConflicts:")

if result["conflicts"]:

    for conflict in result["conflicts"]:
        print("-", conflict)

else:

    print("None")