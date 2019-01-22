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
        for word_id in doc_ids:
            doc.append(self.translate_dict[int(word_id)])
        return doc
