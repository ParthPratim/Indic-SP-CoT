import random
import re

from src.utils import *
from src.api import *
from src.models import *
from tqdm import tqdm
from copy import deepcopy
from typing import Union
import time
import timeit

# TOPICS = {'politicians': 40,
#           'athletes': 20,
#           'sports teams (basketball, soccer, football, baseball .etc)': 20,
#           'sports events (tournaments, leagues, cups .etc)': 20,
#           'countries': 20,
#           'cities': 30,
#           'historical figures': 20,
#           'historical events': 20,
#           'wars': 20,
#           'religions': 10,
#           'singers': 30,
#           'songs': 30,
#           'actors or actresses': 20,
#           'movies or TV series': 20,
#           'writers': 10,
#           'books': 20,
#           'painters': 10,
#           'paintings': 10,
#           'composers': 10,
#           'classical music': 10,
#           'tourist attractions (artificial and natural)': 40,
#           'scientists': 20,
#           'scientific terms': 10,
#           'video games': 10,
#           'animals': 10,
#           'plants': 10,
#           'foods': 10,
#           'enterprises': 20,
#           'international organizations': 20
#           }


TOPICS = {'\u0916\u0947\u0932' : 10, 
'\u0938\u0902\u0917\u0940\u0924' : 10,
'\u0915\u0932\u093e' : 10,
'\u0936\u093f\u0915\u094d\u0937\u093e' : 10,
'\u0935\u093f\u091c\u094d\u091e\u093e\u0928' : 10,
'\u091a\u093f\u0915\u093f\u0924\u094d\u0938\u093e' : 10,
'\u0907\u0924\u093f\u0939\u093e\u0938' : 10,
'\u0930\u093e\u091c\u0928\u0940\u0924\u093f' : 10,
'\u0927\u0930\u094d\u092e' : 10,
'\u0905\u0930\u094d\u0925\u0936\u093e\u0938\u094d\u0924\u094d\u0930' : 10,
'\u0924\u0915\u0928\u0940\u0915' : 10,
'\u0938\u0902\u091a\u093e\u0930' : 10,
'\u092a\u0930\u094d\u092f\u093e\u0935\u0930\u0923' : 10,
'\u092d\u0942\u0917\u094b\u0932' : 10,
'\u0915\u093e\u0928\u0942\u0928\u0940 \u092a\u094d\u0930\u0923\u093e\u0932\u0940' : 10,
'\u0938\u093e\u092e\u093e\u091c\u093f\u0915 \u0935\u094d\u092f\u0935\u0938\u094d\u0925\u093e' : 10,
'\u092f\u0941\u0926\u094d\u0927' : 10,
'\u0935\u094d\u092f\u093e\u092a\u093e\u0930' : 10,
'\u0938\u0902\u0938\u094d\u0915\u0943\u0924\u093f' : 10,
'\u092a\u0930\u094d\u092f\u091f' : 10
          }


def list_topic_terms(model: Union[APIModel, LocalModel], topic: str):
    # prompt = f"List some {topic}, separated by '|':\n"
    prompt = f"List some {topic} separated by '|':"
    print(f"List some {topic} separated by |:")
    response = model.get_response(prompt, temperature=1.0, max_tokens=64)
    print(f"Response: {response}")
    terms = [it.strip() for it in response.split('|')]
    return terms


def generate_passage_for_term(
        model: Union[APIModel, LocalModel],
        term: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        template: Dict = None,
        detailed: bool = False
):
    if template:
        example_term, example_passage = template["term"], template["passage"]
        prompt = f"Generate a passage from Wikipedia about {example_term}:\n{example_passage}\n\n"
    else:
        prompt = ""

    if detailed:
        prompt += f"Generate a detailed passage from Wikipedia about {term}:"
    else:
        prompt += f"Generate a passage from Wikipedia about {term}:"
    raw_passage = model.get_response(prompt, temperature=temperature, max_tokens=max_tokens)

    # remove the incomplete sentence at the end
    raw_sentences = nltk.sent_tokenize(raw_passage)
    sentences = deepcopy(raw_sentences[:-1]) if not raw_sentences[-1].endswith('.') else deepcopy(raw_sentences)
    passage = ' '.join(sentences)

    return passage


def extract_entities_in_passage(
        model: APIModel,
        passage: str,
        max_tokens: int = 128,
        extractor: str = "spacy-nltk",
        ban_entity_list: List[str] = None,
):
    # extract entities
    all_entities = []
    if "spacy" in extractor:
        doc = nlp(passage)
        entities_spacy = [clean_entity(ent.text) for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE']]
        entities_spacy = [ent for ent in entities_spacy if ent is not None]
        all_entities += entities_spacy

    # entity_source_2
    if "llm" in extractor:
        prompt_extract_entity = f"Passage:\n{passage}\n\n" \
                                f"Extract the named entities (like date, location, organization, character, number) in the above passage, entities should be separated by '|'." \
                                f"If no named entity in it, return 'None' only."
        raw_entities_by_llm = model.get_response(prompt_extract_entity, temperature=0.0, max_tokens=max_tokens)

        entities_llm = [clean_entity(ent) for ent in raw_entities_by_llm.split('|')]
        entities_llm = [ent for ent in entities_llm if ent is not None]
        all_entities += entities_llm

    # entity_source_3
    if "nltk" in extractor:
        entities_nltk = process_an_item(passage)
        entities_nltk = [clean_entity(ent) for ent in entities_nltk]
        entities_nltk = [ent for ent in entities_nltk if ent is not None]
        all_entities += entities_nltk

    entities = [[], []]
    for rent in all_entities:
        if rent.endswith('He'):
            rent = rent[:-len('He')]
        if rent.endswith('She'):
            rent = rent[:-len('She')]
        if rent.endswith('They'):
            rent = rent[:-len('They')]
        if rent.endswith('It'):
            rent = rent[:-len('It')]

        if normalize_answer(rent) not in entities[1]:
            entities[0].append(rent)
            entities[1].append(normalize_answer(rent))

    # clean entities[0], so that sub_strings are removed
    final_entities = [[], []]
    for idx, item in enumerate(entities[0]):
        addf = True
        for another_item in entities[0][:idx] + entities[0][idx + 1:]:
            if normalize_answer(item) in normalize_answer(another_item):
                addf = False
                break
        if addf:
            only_lower_case = True
            for char in item:
                if char not in whitespace + ascii_lowercase:
                    only_lower_case = False
                    break
            if only_lower_case:
                final_entities[1].append(item)
            else:
                final_entities[0].append(item)

    random.shuffle(final_entities[0])
    random.shuffle(final_entities[1])

    # 1 are the entities that contain only lowercase, otherwise 0
    final_entities = final_entities[0] + final_entities[1]
    # lemmatize the entities
    lemma_final_entities = [lemmatize(ent) for ent in final_entities]

    output_entities = []
    if ban_entity_list:
        for ent in final_entities:
            ban_flag = False
            for ban_entity in ban_entity_list:
                if normalize_answer(ban_entity) in normalize_answer(ent):
                    ban_flag = True
                    break

                if ban_entity in ent:
                    ban_flag = True
                    break

                if lemmatize(ban_entity) in lemmatize(ent):
                    ban_flag = True
                    break

            if not ban_flag:
                output_entities.append(ent)

    return output_entities


def generate_qae_pair_for_entity(
        model: APIModel,
        passage: str,
        title: str,
        expected_answer: str,
        question_tokens: int = 64,
        answer_tokens: int = 10,
        explanation_tokens: int = 128,
        question_temperature: float = 0.0,
        max_retry: int = 10,
        title_in_question: bool = False,
        template: Dict = None,
        ban_entity_list: List[str] = None,
):
    if template:
        template_passage = template["passage"]
        template_question = template["question"]
        template_answer = template["answer"]
        template_explanation = template["explanation"]
        prompt_generate_question = f"Passage:\n{template_passage}\n\nEntity: {template_answer}\n\n" \
                                   f"Question:\n{template_question}\n\n"
        prompt_generate_answer = f"Passage:\n{template_passage}\n\nQuestion:\n{template_question}\n\nAnswer:\n{template_answer}\n\n"
        prompt_generate_explanation = f"Passage:\n{template_passage}\n\nQuestion:\n{template_question}\n\nAnswer:\n{template_answer}\n\n" \
                                      f"Explanation:\n{template_explanation}\n\n"
    else:
        prompt_generate_question = ""
        prompt_generate_answer = ""
        prompt_generate_explanation = ""

    # generate question
    if title_in_question:
        prompt_generate_question += f"Passage about {title}:\n{passage}\n\n" \
                                    f"Generate a question that meets the following conditions: 1. contains the term '{title}' in question, 2. the answer is '{expected_answer}'."
    else:
        prompt_generate_question += f"Passage about {title}:\n{passage}\n\n" \
                                    f"Generate a question to which the answer is the entity '{expected_answer}'."

    if ban_entity_list:
        prompt_generate_question += f"(Avoid the following entities in the question: {', '.join(ban_entity_list)})."

    question = model.get_response(prompt_generate_question, temperature=question_temperature,
                                  max_tokens=question_tokens,
                                  ban_pronoun=True)

    if title_in_question:
        num_retry = 0
        while not normalize_answer(title.lower()) in normalize_answer(question.lower()):
            if num_retry > max_retry:
                # print(f"Failed to generate question that contains the title: Q=[{question}], title=[{title}].")
                return None
            question = model.get_response(prompt_generate_question, temperature=question_temperature,
                                          max_tokens=question_tokens,
                                          ban_pronoun=True)
            num_retry += 1

    prompt_generate_answer += f"Passage about {title}:\n{passage}\n\nQuestion:\n{question}\n\nExtract the answer directly from the passage in less words as possible."
    raw_answer = model.get_response(prompt_generate_answer, temperature=0.0, max_tokens=answer_tokens)

    num_retry = 0

    while normalize_answer(raw_answer) != normalize_answer(expected_answer):
        if num_retry > max_retry:
            # print(f"Failed to generate expected answer: {expected_answer}, raw answer: {raw_answer}")
            return None
        raw_answer = model.get_response(prompt_generate_answer, temperature=0.0, max_tokens=answer_tokens)
        num_retry += 1

    answer = raw_answer

    # prompt_generate_explanation = f"Passage:\n{passage}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\n" \
    #                            f"Refer to the passage and extract an evidence sentence as explanation, " \
    #                            f"{answer} must in the explanation."

    prompt_generate_explanation += f"Passage about {title}:\n{passage}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\n" \
                                   f"According to an evidence from the passage to support the answer, rewrite it to make its meaning clear without passage."

    # prompt_generate_explanation += f"Passage:\n{passage}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\n" \
    #                                f"According to the passage, write a short explanation for the answer."

    explanation = model.get_response(prompt_generate_explanation, temperature=0.0, max_tokens=explanation_tokens,
                                     ban_pronoun=True)

    # num_retry = 0

    # while answer not in explanation:
    #     explanation = model.get_response(prompt_generate_explanation, temperature=0.5, max_tokens=explanation_tokens,
    #                                      ban_pronoun=True)
    #     if num_retry > max_retry:
    #         print(f"Failed to generate expected explanation: {explanation}, raw answer: {answer}")
    #         return None
    #     num_retry += 1

    return {
        "question": question,
        "answer": answer,
        "explanation": explanation
    }


def generate_qa_pairs_for_passage(
        model: APIModel,
        passage: str,
        title: str = "",
        entity_tokens: int = 64,
        question_tokens: int = 64,
        answer_tokens: int = 10,
        explanation_tokens: int = 128,
        extractor: str = "nltk-spacy",
        max_qae_pairs: int = 10
):
    # extract entities
    entities = extract_entities_in_passage(model, passage, max_tokens=entity_tokens, extractor=extractor)

    # generate question-answer pairs
    qae_pairs = []
    normalized_questions = []
    for entity in entities:
        if len(entity.split()) > 5:
            continue

        qae_pair = generate_qae_pair_for_entity(
            model=model,
            passage=passage,
            title=title,
            expected_answer=entity,
            question_tokens=question_tokens,
            answer_tokens=answer_tokens,
            explanation_tokens=explanation_tokens
        )
        if qae_pair is None:
            continue

        duplicated_question = False
        for q in normalized_questions:
            if normalize_answer(qae_pair["question"]) == q:
                duplicated_question = True
                break
        if duplicated_question:
            continue
        else:
            normalized_questions.append(normalize_answer(qae_pair["question"]))
            qae_pairs.append(qae_pair)

        if len(qae_pairs) >= max_qae_pairs:
            break

    return qae_pairs


def generate_examples_for_topic(
        model: APIModel,
        topic: str,
        num_terms: int,
        passage_tokens: int = 256,
        entity_tokens: int = 64,
        question_tokens: int = 64,
        answer_tokens: int = 10,
        explanation_tokens: int = 128,
        extractor: str = "nltk-spacy",
        max_qae_pairs: int = 10,
        topic_cate: str = None,
):
    # collect terms
    all_terms = []
    while len(all_terms) < num_terms:
        terms = list_topic_terms(model, topic)
        for term in terms:
            if term not in all_terms:
                all_terms.append(term)
        print(f"{topic} has collected {len(all_terms)} terms.")

    all_terms = all_terms[:num_terms]
    print(f"{topic} has collected {len(all_terms)} terms, Done.")

    topic_qa_pairs = []

    # generate passages
    for term in tqdm(all_terms, desc=f"Generating passages for {topic}"):
        passage = generate_passage_for_term(
            model,
            term=term,
            max_tokens=passage_tokens,
        )

        # generate question-answer pairs
        qae_pairs = generate_qa_pairs_for_passage(
            model,
            passage=passage,
            title=term,
            entity_tokens=entity_tokens,
            question_tokens=question_tokens,
            answer_tokens=answer_tokens,
            explanation_tokens=explanation_tokens,
            extractor=extractor,
            max_qae_pairs=max_qae_pairs
        )

        for qae_pair in qae_pairs:
            topic_qa_pairs.append({
                "topic": topic,
                "term": term,
                "passage": passage,
                "question": qae_pair["question"],
                "answer": qae_pair["answer"],
                "explanation": qae_pair["explanation"]
            })

    return topic_qa_pairs


# self prompt cot code
def generate_base_quadruples(
        model: APIModel,
        term: str,
        passage_tokens: int = 512,
        question_tokens: int = 64,
        answer_tokens: int = 10,
        explanation_tokens: int = 128,
        num_questions_per_paragraph: int = 2,
        entity_extractor: str = "nltk-spacy"
):
    # generate long and detailed passage
    passage = generate_passage_for_term(
        model=model,
        term=term,
        max_tokens=passage_tokens,
        detailed=True
    )

    paragraphs = []
    sentences = nltk.sent_tokenize(passage)

    for i in range(0, len(sentences), 2):
        paragraphs.append(" ".join(sentences[i:i + 2]))
    if len(sentences) % 2 != 0:
        paragraphs[-1] = paragraphs[-1] + " " + sentences[-1]
    existing_questions = []
    existing_answers = []
    output = {
        "term": term,
        "passage": passage,
        "qae_pairs": []
    }

    # generate question-answer pairs for each paragraph
    for paragraph in tqdm(paragraphs):
        num_retry = 0
        qae_pairs = []

        entities = extract_entities_in_passage(
            model=model,
            passage=paragraph,
            max_tokens=64,
            extractor=entity_extractor,
        )

        for entity in entities:
            qae_pair = generate_qae_pair_for_entity(
                model=model,
                passage=paragraph,
                title=term,
                expected_answer=entity,
                question_tokens=question_tokens,
                answer_tokens=answer_tokens,
                explanation_tokens=explanation_tokens
            )
            if qae_pair is None:
                continue

            if normalize_answer(qae_pair["question"]) in existing_questions:
                continue

            if normalize_answer(qae_pair["answer"]) in existing_answers:
                continue

            existing_questions.append(normalize_answer(qae_pair["question"]))
            existing_answers.append(normalize_answer(qae_pair["answer"]))
            qae_pairs.append(qae_pair)

            if len(qae_pairs) >= num_questions_per_paragraph:
                break

        for qae_pair in qae_pairs:
            output["qae_pairs"].append({
                "context": paragraph,
                "question": qae_pair["question"],
                "answer": qae_pair["answer"],
                "explanation": qae_pair["explanation"]
            })

    return output


def extract_entities_and_labels_spacy(
        passage: str,
        ban_entity_list: List[str] = None,
):
    doc = nlp(passage)
    entities = [clean_entity(ent.text) for ent in doc.ents]
    labels = [ent.label_ for ent in doc.ents]

    if ban_entity_list:
        final_entities, final_labels = [], []
        lemma_final_entities = [lemmatize(ent) for ent in entities]
        lemma_ban_entity_list = [lemmatize(ent) for ent in ban_entity_list]
        block_entities = [normalize_answer(ent) for ent in lemma_ban_entity_list]
        for i in range(len(entities)):
            if normalize_answer(lemma_final_entities[i]) not in block_entities:
                final_entities.append(entities[i])
                final_labels.append(labels[i])
        entities = final_entities
        labels = final_labels

    return entities, labels


def generate_multi_hop_quadruples(
        model: APIModel,
        term: str,
        passage_tokens: int = 512,
        question_tokens: int = 64,
        answer_tokens: int = 10,
        explanation_tokens: int = 128,
        num_entity_per_paragraph: int = 2,
        min_passage_tokens: int = 128,
        entity_extractor: str = "spacy",
        question_ban_list: List[str] = None,
        answer_ban_list: List[str] = None,
        detailed: bool = False,
        title_in_question: bool = False,
):
    # filter out terms like November 2020
    if re.match(DATE_PATTERN, term.lower()) is not None:
        # print(f"Term {term} is a date, skip it.")
        return None

    # filter out terms that are numbers
    if re.match(r"\d+$", normalize_answer(term)) is not None:
        # print(f"Term {term} is a number, skip it.")
        return None

    for e in ["million", "trillion", "billion", "hundred", "thousand"]:
        if e in term:
            return None

    # refuse to generate passage for [QUANTITY, ORDINAL, CARDINAL, PERCENT, MONEY, DATE, TIME]
    input_entities = nlp(term)
    for ent in input_entities.ents:
        if ent.label_ in ["QUANTITY", "ORDINAL", "CARDINAL", "PERCENT", "MONEY", "DATE", "TIME"]:
            # print(f"Term {term} is a {ent.label_}, skip it.")
            return None

    # generate long and detailed passage
    passage = generate_passage_for_term(
        model=model,
        term=term,
        max_tokens=passage_tokens,
        detailed=detailed
    )

    if len(passage.split()) < min_passage_tokens:
        # print(f"Passage for {term} is too short, skip it. (len: {len(passage.split())})")
        return None

    paragraphs = []
    sentences = nltk.sent_tokenize(passage)

    for i in range(0, len(sentences), 2):
        paragraphs.append(" ".join(sentences[i:i + 2]))
    if len(sentences) % 2 != 0:
        paragraphs[-1] = paragraphs[-1] + " " + sentences[-1]

    existing_questions = []
    existing_answers = []
    existing_invalid_answers = []
    output = {
        "term": term,
        "passage": passage,
        "qae_pairs": []
    }

    # generate question-answer pairs for each paragraph
    # for paragraph in tqdm(paragraphs, desc=f"Generate QA pairs for {term}"):
    print(f"Generate QA pairs for {term}.......")
    for i, paragraph in enumerate(paragraphs):
        print(f"\tProgress: {i + 1}/{len(paragraphs)}")
        qae_pairs = []

        # We should avoid generating answers for the same entity in previous answers
        entities = extract_entities_in_passage(
            model=model,
            passage=paragraph,
            extractor=entity_extractor,
            ban_entity_list=answer_ban_list
        )
        #
        # # entities = [ent for ent in entities if ent not in term]
        # entities, labels = extract_entities_and_labels_spacy(
        #     passage=paragraph,
        #     ban_entity_list=answer_ban_list
        # )

        random.shuffle(entities)

        processed_cnt = 0
        for entity in entities:
            if entity in existing_invalid_answers:
                continue
            if processed_cnt >= num_entity_per_paragraph:
                break
            processed_cnt += 1
            # We should avoid generating questions containing previous answers in question text
            t1 = time.time()
            qae_pair = generate_qae_pair_for_entity(
                model=model,
                passage=paragraph,
                title=term,
                expected_answer=entity,
                question_tokens=question_tokens,
                answer_tokens=answer_tokens,
                explanation_tokens=explanation_tokens,
                ban_entity_list=question_ban_list,
                title_in_question=title_in_question
            )
            t2 = time.time()

            if qae_pair is None:
                # print(f"Generate qae pair for {entity} takes {t2 - t1} seconds. False.")
                existing_invalid_answers.append(entity)
                continue
            # print(f"Generate qae pair for {entity} takes {t2 - t1} seconds.True.")

            if normalize_answer(qae_pair["question"]) in existing_questions:
                continue

            if normalize_answer(qae_pair["answer"]) in existing_answers:
                existing_invalid_answers.append(entity)
                continue

            existing_questions.append(normalize_answer(qae_pair["question"]))
            existing_answers.append(normalize_answer(qae_pair["answer"]))
            qae_pairs.append(qae_pair)

        for qae_pair in qae_pairs:
            output["qae_pairs"].append({
                "context": paragraph,
                "question": qae_pair["question"],
                "answer": qae_pair["answer"],
                "explanation": qae_pair["explanation"]
            })

    return output


def generate_quadruples_for_term(
        model: APIModel,
        term: str,
        passage_tokens: int = 512,
        question_tokens: int = 64,
        answer_tokens: int = 10,
        explanation_tokens: int = 128,
        num_questions_per_paragraph: int = 1,
        max_retry: int = 10,
):
    # generate long and detailed passage
    passage = generate_passage_for_term(
        model=model,
        term=term,
        max_tokens=passage_tokens,
        detailed=True
    )

    paragraphs = []
    sentences = nltk.sent_tokenize(passage)

    for i in range(0, len(sentences), 2):
        paragraphs.append(" ".join(sentences[i:i + 2]))
    if len(sentences) % 2 != 0:
        paragraphs[-1] = paragraphs[-1] + " " + sentences[-1]
    existing_questions = []
    output = {
        "term": term,
        "passage": passage,
        "qae_pairs": []
    }

    # generate question-answer pairs for each paragraph
    for paragraph in tqdm(paragraphs):
        num_retry = 0
        qae_pairs = []
        while len(qae_pairs) < num_questions_per_paragraph and num_retry < max_retry:
            qae_pair = generate_qae_pair_for_entity(
                model=model,
                passage=paragraph,
                title=term,
                expected_answer=term,
                question_tokens=question_tokens,
                answer_tokens=answer_tokens,
                explanation_tokens=explanation_tokens
            )
            if qae_pair is None:
                num_retry += 1
                continue

            if normalize_answer(qae_pair["question"]) not in existing_questions:
                existing_questions.append(normalize_answer(qae_pair["question"]))
                qae_pairs.append(qae_pair)
            else:
                num_retry += 1

            if num_retry > 10:
                break

        for qae_pair in qae_pairs:
            output["qae_pairs"].append({
                "context": paragraph,
                "question": qae_pair["question"],
                "answer": qae_pair["answer"],
                "explanation": qae_pair["explanation"]
            })

    return output


#
# def generate_hop_quadruples_for_question(
#         model: APIModel,
#         base_answer: str,
#         base_question: str,
#         passage_tokens: int = 512,
#         question_tokens: int = 64,
#         answer_tokens: int = 10,
#         explanation_tokens: int = 128,
#         num_questions_per_paragraph: int = 1,
#         max_retry: int = 10,
#         ban_entity_list: List[str] = None,
#         ban_entity_types: List[str] = None,
# ):
#
#     # generate long and detailed passage
#     passage = generate_passage_for_term(
#         model=model,
#         term=base_answer,
#         max_tokens=passage_tokens,
#         detailed=False
#     )
#
#     raw_paragraphs = []
#     sentences = nltk.sent_tokenize(passage)
#
#     for i in range(0, len(sentences), 2):
#         raw_paragraphs.append(" ".join(sentences[i:i + 2]))
#     if len(sentences) % 2 != 0:
#         raw_paragraphs[-1] = raw_paragraphs[-1] + " " + sentences[-1]
#
#     existing_questions = [normalize_answer(base_question)]
#
#     output = {
#         "term": base_answer,
#         "passage": passage,
#         "qae_pairs": []
#     }
#
#     # generate question-answer pairs for each paragraph
#     for paragraph in tqdm(raw_paragraphs):
#         num_retry = 0
#         qae_pairs = []
#         while len(qae_pairs) < num_questions_per_paragraph and num_retry < max_retry:
#             qae_pair = generate_qae_pair_for_entity(
#                 model=model,
#                 passage=paragraph,
#                 title=base_answer,
#                 expected_answer=term,
#                 question_tokens=question_tokens,
#                 answer_tokens=answer_tokens,
#                 explanation_tokens=explanation_tokens,
#                 ban_entity_list=q_entity
#             )
#             if qae_pair is None:
#                 num_retry += 1
#                 continue
#
#             if normalize_answer(qae_pair["question"]) not in existing_questions:
#                 existing_questions.append(normalize_answer(qae_pair["question"]))
#                 qae_pairs.append(qae_pair)
#             else:
#                 num_retry += 1
#
#             if num_retry > 10:
#                 break
#
#         for qae_pair in qae_pairs:
#             output["qae_pairs"].append({
#                 "context": paragraph,
#                 "question": qae_pair["question"],
#                 "answer": qae_pair["answer"],
#                 "explanation": qae_pair["explanation"]
#             })
#
#     return output
#
#


def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    # Default parameters
    parser.add_argument("--task", type=str, default="self-prompt-cot",
                        choices=["self-prompt", "self-prompt-cot"],
                        help="task name in [open-domain-qa, fact-verification]")

    # parser.add_argument("--method", type=str, default="2-hop",
    #                     choices=["2-hop", "3-hop-linear", "3-hop-tree", "4-hop-linear", "4-hop-tree-balance", "4-hop-tree-unbalance"],
    #                     help="method name in [2-hop, 3-hop-linear, 3-hop-tree, 4-hop-linear, 4-hop-tree-balance, 4-hop-tree-unbalance]")

    parser.add_argument(
        "--model_name", type=str, default="gpt-3.5-turbo-0301", help="model used for response generation.")

    parser.add_argument("--output_path", type=str, default="data/self-prompt-cot/")
    parser.add_argument("--output_file", type=str, default=None)

    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument(
        "--passage_tokens", type=int, default=512, help="maximum length of output tokens by model for zero-shot")
    parser.add_argument(
        "--question_tokens", type=int, default=64, help="maximum length of output tokens by model for zero-shot")
    parser.add_argument(
        "--answer_tokens", type=int, default=10, help="maximum length of output tokens by model for zero-shot")
    parser.add_argument(
        "--explanation_tokens", type=int, default=128, help="maximum length of output tokens by model for zero-shot")
    parser.add_argument(
        "--min_passage_tokens", type=int, default=80, help="maximum length of output tokens by model for zero-shot")

    parser.add_argument(
        "--topic", type=str, default="politician", help="topic name for zero-shot")

    parser.add_argument(
        "--terms_per_topic", type=int, default=10, help="number of terms per topic for zero-shot")

    parser.add_argument(
        "--num_qae_pairs_per_term", type=str, default=5, help="number of qae pairs per term for zero-shot")

    parser.add_argument("--max_terms", type=int, default=10, help="maximum number of terms to generate")

    parser.add_argument("--topic_index", type=int, default=-1, help="index of topic to generate")
    parser.add_argument("--log", type=str, default="", help="index of topic to generate")
    parser.add_argument("--log_dir", type=str, default="", help="index of topic to generate")

    args = parser.parse_args()

    if args.topic_index >= 0:
        args.topic = list(TOPICS.keys())[args.topic_index]
        args.max_terms = TOPICS[args.topic]
        print(f"topic: {args.topic}, max_terms: {args.max_terms}")

    # get base dir of args.output_path
    args.output_path = os.path.join(args.output_path, args.model_name)
    os.makedirs(args.output_path, exist_ok=True)
    output_file_name = f"{args.topic}_{args.max_terms}.json"
    args.output_file = os.path.join(args.output_path, output_file_name)

    if args.log_dir == "":
        args.log_dir = f"logs/{args.model_name}"
    os.makedirs(args.log_dir, exist_ok=True)
    args.log = os.path.join(args.log_dir, f"{args.topic}_{args.max_terms}.log")

    print(f"output file: {args.output_file}")
    print(f"log file: {args.log}")

    return args


def main(args):
    fix_seed(args.random_seed)
    logger = open(args.log, "a")

    # init model
    if args.model_name in ["gpt-3.5-turbo-0301", "gpt-3.5-turbo"]:
        model = ChatGPT(model_name=args.model_name)
    elif args.model_name in ["text-davinci-002", "text-davinci-003"]:
        model = CompleteGPT(model_name=args.model_name)
    elif args.model_name in ["gpt-neoxt-chat"]:
        model = GPTNeoXTChat(model_name=args.model_name)
    elif args.model_name in ["flan-ul2", "flan-t5-xxl"]:
        model = Flan_UL2_20B(model_name=args.model_name)
    elif args.model_name in [
        "t5-11b-ssm", "t5-xxl-ssm", "t5-large-ssm"
    ]:
        model = T5_SSM(model_name=args.model_name)
    elif args.model_name in ["alpaca-7b"]:
        model = Alpaca(model_name=args.model_name)
    elif args.model_name in ["falcon-7b"]:
        model = Falcon(model_name=args.model_name)
    else:
        raise ValueError(f"model {args.model_name} not supported.")

    all_terms = []
    while len(all_terms) < args.max_terms:
        terms = list_topic_terms(model, args.topic)
        print(f"collected {len(all_terms)}/ {args.max_terms}")
        for term in terms:
            if term not in all_terms:
                all_terms.append(term)
            if len(all_terms) >= args.max_terms:
                break
    print(f"collected {len(all_terms)} terms: {all_terms}")
    logger.write(f"collected {len(all_terms)} terms: {all_terms}\n")

    # generate base quadruples
    first_hops = {}
    second_hops = {}
    dataset = {}

    print("========First Hop=========")
    logger.write("========First Hop=========\n")
    print_now()
    for i, term in enumerate(all_terms):
        print(f"[{print_now(1)}] Processing term [{term}], progress {i + 1}/{len(all_terms)}")
        logger.write(f"[{print_now(1)}] Processing term [{term}], progress {i + 1}/{len(all_terms)}\n")
        output = generate_multi_hop_quadruples(
            model=model,
            term=term,
            passage_tokens=args.passage_tokens,
            question_tokens=args.question_tokens,
            answer_tokens=args.answer_tokens,
            explanation_tokens=args.explanation_tokens,
            min_passage_tokens=args.min_passage_tokens,
            num_entity_per_paragraph=3,
            answer_ban_list=[term],
            detailed=True,
            title_in_question=False
        )
        if output and len(output["qae_pairs"]) > 0:
            output["hops"] = [term]
            first_hops[term] = output

    dataset["first_hops"] = first_hops
    with open(args.output_file, "w") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print("\n\n\n========Second Hop=========")
    logger.write("\n\n\n========Second Hop=========\n")
    print_now()
    for j, term in enumerate(first_hops):
        print(f"[{print_now(1)}] Processing [{term}] qae_pairs, progress {j}/{len(first_hops)}...")
        logger.write(f"[{print_now(1)}] Processing [{term}] qae_pairs, progress {j}/{len(first_hops)}...\n")
        for i, qae_pair in enumerate(first_hops[term]["qae_pairs"]):
            print(
                f"[{print_now(1)}] generating second hop for {qae_pair['answer']}, progress {i}/{len(first_hops[term]['qae_pairs'])}...")
            logger.write(
                f"[{print_now(1)}] generating second hop for {qae_pair['answer']}, progress {i}/{len(first_hops[term]['qae_pairs'])}...\n")
            if qae_pair["answer"] in first_hops:
                output = first_hops[qae_pair["answer"]]
            else:
                output = generate_multi_hop_quadruples(
                    model=model,
                    term=qae_pair["answer"],
                    passage_tokens=args.passage_tokens,
                    question_tokens=args.question_tokens,
                    answer_tokens=args.answer_tokens,
                    explanation_tokens=args.explanation_tokens,
                    min_passage_tokens=args.min_passage_tokens,
                    num_entity_per_paragraph=3,
                    answer_ban_list=first_hops[term]["hops"] + [qae_pair["answer"]],
                    question_ban_list=first_hops[term]["hops"],
                    detailed=False,
                    title_in_question=True
                )
            if output and len(output["qae_pairs"]) > 0:
                output["hops"] = first_hops[term]["hops"] + [qae_pair["answer"]]
                second_hops[qae_pair["answer"]] = output
            # else:
            #     print(f"failed to generate second hop for {qae_pair['answer']}")

    dataset["second_hops"] = second_hops
    with open(args.output_file, "w") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print("\n\n\n========End=========")
    logger.write("\n\n\n========End=========\n")
    logger.close()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
    # print("Done!")

