import os
import json
dir = "data/dir"
with open("/home/p.zak2/PycharmProjects/ner_poleval/data/test/poleval_test_ner_2018.json") as f:
    main_data = json.load(f)
for file in os.listdir(dir):
    if file.endswith(".json") and file.startswith("test") and not file.endswith("fixed.json"):
        with open(dir+"/"+file) as f:
            data  = json.load(f)
            prev_text = ""
            for idx, doc in enumerate(data):
                ents = {}
                answers = doc["answers"].split("\n")
                new_labels = []
                for answer in answers:
                    if answer!= "":
                        val = answer.split("\t")
                        if  len(val) > 1:
                            entity, text = val[0],  val[1]
                        else:
                            print("Error {}".format(val))
                            continue
                        category, start, end = entity.split(" ")
                        if category == 'P' or category == 'O':
                            print("Padding or outside {}".format(val))
                            continue
                        new_text = main_data[idx]["text"][int(start):int(end)]
                        if new_text == prev_text:
                            continue
                        prev_text = new_text
                        new_labels.append("{} {} {}\t{}".format(category, start, end,new_text))
                new_labels_str = "\n".join(new_labels)
                doc["text"] = main_data[idx]["text"]
                doc["answers"] = new_labels_str
        with open(dir+"/"+file.split(".")[0]+"2fixed.json", "w+") as f:
            json.dump(data,f, indent=2)