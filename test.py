import json

contexts = []

with open("xquad.hi.json", "r") as f:
    objs = json.load(f)['data']
    for entry in objs:
        for section in entry["paragraphs"]:
            contexts.append(section['context'] + "\n")
            contexts.append("\n")
lines = []

lines.append("Below are some hindi paragraphs providing context. Extract broad 18-20 hindi categories and topics that the below paragraphs lie in. Ensure that 60 percent of the topics are single word. Remanining can be little but not very long.\n")
lines.extend(contexts)

with open("/home/parth/courses/FML/prompts.txt", "w") as f:
    f.writelines(lines)

