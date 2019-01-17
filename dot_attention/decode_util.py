from nltk.translate.bleu_score import sentence_bleu
class Evaluate:
    def __init__(self, target_dict, val=0 ,gold_sentence_file=None, val_iter=None):
        self.target_dict = target_dict
        if val:
            self.val_iter = val_iter
            self.gold_sentence = self.GetGoldSentence(gold_sentence_file)
        self.translate_dict = self.GetTranslateDict(target_dict)

    def GetGoldSentence(self, gold_sentence_file):
        gold_sentence = []
        with open(gold_sentence_file) as lines:
            for line in lines:
                gold_sentence.append(line.split())
        return gold_sentence

    def GetTranslateDict(self, target_dict):
        translate_dict = {}
        for key, value in target_dict.items():
            translate_dict[value] = key
        return translate_dict

    def TranslateSentence(self, sentence_ids):
        sentence = []
        for sentence_id in sentence_ids:
            sentence.append(self.translate_dict[int(sentence_id)])
        return sentence