import os
import json
dir = "test"
counts = 0
all_counts =0
with open("POLEVAL-NER_GOLD.json") as f:
    main_data = json.load(f)["questions"]
    for idx, doc in enumerate(main_data):
        answers = doc["answers"][0]["data"]["brat"].split("\n")
        new_labels = []
        ents = []
        for answer in answers:
            if answer!= "":
                val = answer.split("\t")
                if  len(val) > 1:
                    entity, text = val[0],  val[1]
                else:
                    print("Error {}".format(val))
                    continue
                try:
                    category, start, end = text.split(" ")
                    for x in ents:
                        if x["category"][:3] != category[:3]:
                            if not( end < x["start"] or x["end"] < start) and start.isnumeric():
                                counts+=1
                                break
                    all_counts+=1
                    ents.append(({"start":start, "end":end, "category":category}))
                except:
                    pass
print(counts)
print(all_counts)