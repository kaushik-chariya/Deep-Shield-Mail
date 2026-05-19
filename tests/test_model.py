def _transform(self, samples):
    if isinstance(self.preprocessor, dict):
        # Replace "vectorizer" with the actual key from debug output
        vectorizer = self.preprocessor.get("vectorizer") \
                  or self.preprocessor.get("tfidf") \
                  or list(self.preprocessor.values())[0]
        return vectorizer.transform(samples)
    return self.preprocessor.transform(samples)