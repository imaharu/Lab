class GenerateUtil:
    def __init__(self, target_dict):
        self.target_dict = target_dict
        self.translate_dict = self.GetTranslateDict(target_dict)

    def GetTranslateDict(self, target_dict):
        translate_dict = {}
        for key, value in target_dict.items():
            translate_dict[value] = key
        return translate_dict

    def TranslateDoc(self, doc_ids):
        doc = []
        for sentence_ids in doc_ids:
            sentence = []
            for word_id in sentence_ids:
                sentence.append(self.translate_dict[word_id])
            doc.append(" ".join(sentence))
        return doc
