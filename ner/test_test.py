import json
test_data = [{
'text':'Miasto postanowiło za jednym zamachem trzy spółki połączyć w jedną. Przygotowany jest projekt uchwały, który przewiduje wniesienie udziałów w Towarzystwie Budownictwa Społecznego \"Wielkopolska\" oraz Towarzystwie Budownictwa Społecznego \"Nasz Dom\" do Poznańskiego Towarzystwa Budownictwa Społecznego. W piątek opiniować tę propozycję będzie Komisja Gospodarki Komunalnej i Polityki Mieszkaniowej, a we wtorek zajmie się nią Rada Miasta.\n\n- Pomysł połączenia TBS-ów nie budzi wątpliwości z punktu widzenia racjonalizacji kosztów - twierdzi Tomasz Lewandowski, radny LiD i członek komisji. - Potrzebna jest jednak dyskusja o przyszłości towarzystw. Obecnie rząd pracuje nad zmianą ustawy, która przewiduje wykup mieszkań w towarzystwach budownictwa społecznego. To stworzy zupełnie nową sytuację. W związku z tym konieczne będzie podjęcie odpowiednich kroków przez miasto.\n\nNorbert Napieraj, szef klubu radnych PiS również uważa, że ze względów ekonomicznych utworzenie jednej spółki jest zasadne.\n\n- Na razie jest to jednak luźny pomysł. Nie ma konkretów - dodaje N. Napieraj. - Nasz klub jeszcze nie wypracował w sprawie tej uchwały stanowiska.',
'id':'PCCwR-1.1-TXT/very_short/Dzienniki/1b.txt'}]

data = [{
'text':'Miasto postanowiło za jednym zamachem trzy spółki połączyć w jedną. Przygotowany jest projekt uchwały, który przewiduje wniesienie udziałów w Towarzystwie Budownictwa Społecznego \"Wielkopolska\" oraz Towarzystwie Budownictwa Społecznego \"Nasz Dom\" do Poznańskiego Towarzystwa Budownictwa Społecznego. W piątek opiniować tę propozycję będzie Komisja Gospodarki Komunalnej i Polityki Mieszkaniowej, a we wtorek zajmie się nią Rada Miasta.\n\n- Pomysł połączenia TBS-ów nie budzi wątpliwości z punktu widzenia racjonalizacji kosztów - twierdzi Tomasz Lewandowski, radny LiD i członek komisji. - Potrzebna jest jednak dyskusja o przyszłości towarzystw. Obecnie rząd pracuje nad zmianą ustawy, która przewiduje wykup mieszkań w towarzystwach budownictwa społecznego. To stworzy zupełnie nową sytuację. W związku z tym konieczne będzie podjęcie odpowiednich kroków przez miasto.\n\nNorbert Napieraj, szef klubu radnych PiS również uważa, że ze względów ekonomicznych utworzenie jednej spółki jest zasadne.\n\n- Na razie jest to jednak luźny pomysł. Nie ma konkretów - dodaje N. Napieraj. - Nasz klub jeszcze nie wypracował w sprawie tej uchwały stanowiska.',
'id':'PCCwR-1.1-TXT/very_short/Dzienniki/1b.txt',
'answers': 'orgName 142 193\tTowarzystwie Budownictwa Społecznego \"Wielkopolska\"\norgName 199 246\tTowarzystwie Budownictwa Społecznego \"Nasz Dom\"\nplaceName_settlement 250 262\tPoznańskiego\norgName 250 298\tPoznańskiego Towarzystwa Budownictwa Społecznego\norgName 340 394\tKomisja Gospodarki Komunalnej i Polityki Mieszkaniowej\norgName 423 434\tRada Miasta\npersName 538 556\tTomasz Lewandowski\npersName_forename 538 544\tTomasz\npersName_surname 545 556\tLewandowski\norgName 564 567\tLiD\npersName 871 887\tNorbert Napieraj\npersName_forename 871 878\tNorbert\npersName_surname 879 887\tNapieraj\norgName 908 911\tPiS\npersName 1062 1073\tN. Napieraj\npersName_forename 1062 1064\tN.\npersName_surname 1065 1073\tNapieraj\nplaceName_region 180 192\tWielkopolska\nderivType_placeName_settlement 250 262\tPoznańskiego'
}]
for test, my_test in zip(test_data, data):
    # assert test["text"] == my_test["text"]
    fixed_answers = []
    answers = my_test["answers"].split("\n")
    for idx, answer in enumerate(answers):
        if answer and len(answer.split("\t")) ==2:
            entity, text = answer.split("\t")
            if len(text) < 3:
                print(text)
            category, start,end = entity.split(" ")
            if  text != test["text"][int(start):int(end)]:
                print(text + " : "+ test["text"][int(start):int(end)])
            assert text == test["text"][int(start):int(start)+len(text)]
            fixed_answers.append("{} {} {}\t{}".format(category, start, int(start)+len(text), test["text"][int(start):int(start)+len(text)]))
    assert data[0]["answers"] == "\n".join(fixed_answers)
    my_test["answers"] = "\n".join(fixed_answers)

with open("prediction_poleval_test77.json","w") as f:
    json.dump(data,f,indent=2)

