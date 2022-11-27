import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering

import numpy as np
from constants import *

class QuestionAnsweringModel(object):
    def __init__(self) -> None:
        self.__load_bert_base_multilingual_model__()
        self.__load_xlm_roberta_base_model__()
        self.__load_xlm_roberta_large_model__()

    def __load_bert_base_multilingual_model__(self):
        self.bert_base_multilingual_maxlen = 512
        self.bert_base_multilingual_stride = 128

        self.bert_base_multilingual_tokenizer = AutoTokenizer.from_pretrained(MODEL_BERT_BASE_MULTILINGUAL, use_fast=True)

        self.bert_base_multilingual_model = TFAutoModelForQuestionAnswering.from_pretrained(MODEL_BERT_BASE_MULTILINGUAL)
        self.bert_base_multilingual_model.load_weights(f"resources/{MODEL_BERT_BASE_MULTILINGUAL}")

    def __load_xlm_roberta_base_model__(self):
        self.xlm_roberta_base_maxlen = 512
        self.xlm_roberta_base_stride = 128

        self.xlm_roberta_base_tokenizer = AutoTokenizer.from_pretrained(MODEL_XLM_ROBERTA_BASE, use_fast=True)
        
        self.xlm_roberta_base_model = TFAutoModelForQuestionAnswering.from_pretrained(MODEL_XLM_ROBERTA_BASE)
        self.xlm_roberta_base_model.load_weights(f"resources/{MODEL_XLM_ROBERTA_BASE}")

    def __load_xlm_roberta_large_model__(self):
        self.xlm_roberta_large_maxlen = 512
        self.xlm_roberta_large_stride = 128

        self.xlm_roberta_large_tokenizer = AutoTokenizer.from_pretrained(MODEL_XLM_ROBERTA_LARGE, use_fast=True)
        
        self.xlm_roberta_large_model = TFAutoModelForQuestionAnswering.from_pretrained(MODEL_XLM_ROBERTA_LARGE)
        self.xlm_roberta_large_model.load_weights(f"resources/{MODEL_XLM_ROBERTA_LARGE}")

    def tokenize_question_context(self, question, context, model_name):
        question = question.strip()
        context = context.strip()
        
        if model_name == MODEL_BERT_BASE_MULTILINGUAL:
            tokenizer = self.bert_base_multilingual_tokenizer
            maxlen = self.bert_base_multilingual_maxlen
            stride = self.bert_base_multilingual_stride
        elif model_name == MODEL_XLM_ROBERTA_BASE:
            tokenizer = self.xlm_roberta_base_tokenizer
            maxlen = self.xlm_roberta_base_maxlen
            stride = self.xlm_roberta_base_stride
        elif model_name == MODEL_XLM_ROBERTA_LARGE:
            tokenizer = self.xlm_roberta_large_tokenizer
            maxlen = self.xlm_roberta_large_maxlen
            stride = self.xlm_roberta_large_stride

        inputs = tokenizer(
            question,
            context,
            max_length=maxlen,
            truncation="only_second",
            stride=stride,
            padding="max_length",
            return_tensors="tf"
        )

        return inputs

    def question_answer(self, question, context, model_name):
        if model_name == MODEL_BERT_BASE_MULTILINGUAL:
            model = self.bert_base_multilingual_model
            tokenizer = self.bert_base_multilingual_tokenizer
        elif model_name == MODEL_XLM_ROBERTA_BASE:
            model = self.xlm_roberta_base_model
            tokenizer = self.xlm_roberta_base_tokenizer
        elif model_name == MODEL_XLM_ROBERTA_LARGE:
            model = self.xlm_roberta_large_model
            tokenizer = self.xlm_roberta_large_tokenizer

        inputs = self.tokenize_question_context(question, context, model_name)
        outputs = model(**inputs)

        start_logits = outputs["start_logits"].numpy()
        end_logits = outputs["end_logits"].numpy()

        starts = np.argmax(start_logits, axis=1)
        ends = np.argmax(end_logits,  axis=1)

        start_scores = np.max(start_logits, axis=1)
        end_scores = np.max(end_logits, axis=1)
        scores = start_scores + end_scores

        indices = []
        for idx, start in enumerate(starts):
            end = ends[idx]
            if start == 0 and end == 0:
                continue
            if end < start:
                continue
            indices.append(idx)

        answers = []
        for idx in indices:
            score = scores[idx]
            ans_ids = inputs["input_ids"][idx][starts[idx]:ends[idx]+1]
            answer = tokenizer.decode(ans_ids, skip_special_tokens=True)
            answers.append((answer, score))

        return answers
