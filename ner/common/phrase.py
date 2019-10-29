class Phrase:
    def __init__(self, start_idx):
        self.start_idx = start_idx
        self.end_idx = start_idx
        self.text = ""
        self.start = 0
        self.end = 0
        self.category = "None"

    def __repr__(self):
        return "T\t{} {} {}\t{}".format(self.category,self.start,self.end,self.text)

    def __str__(self):
        return "T\t{} {} {}\t{}".format(self.category,self.start,self.end,self.text)
