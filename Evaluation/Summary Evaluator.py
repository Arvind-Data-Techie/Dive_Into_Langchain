from typing import List
from langsmith.schemas import Example, Run
from langsmith.evaluation import evaluate

from langchain. prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
# LLM
llm = ChatOllama(model="mistral", format="json", temperature=0)

prompt=PromptTemplate(
template='''You are a grader assessing relevance of a retrieved document to a user question. \n
    Here is the retrieved document: \n\n {document} \n\n\
Here is the user question: {question} \n
If the document contains keywords related to the user question, grade it as relevant. \n
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
Provide the binary score as a JSON with a single key 'score' and no premable or explaination.''',
input_variables= ["question", "document"])


retrieval_grader_mistral = prompt | llm | JsonOutputParser()

def predict_mistral (inputs: dict) -> dict:
    grade = retrieval_grader_mistral.invoke({"question": inputs ["question"], "document": inputs ["doc_txt"]})
    return {"grade": grade ['score']}

def f1_score_summary_evaluator(runs: List[Run], examples: List[Example]) -> dict:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for run, example in zip(runs, examples):
        # Matches the output format of your dataset
        reference = example.outputs["answer"]
        # Matches the output dict in `predict` function below
        prediction = run.outputs["prediction"]
        if reference and prediction == reference:
            true_positives += 1
        elif prediction and not reference:
            false_positives += 1
        elif not prediction and reference:
            false_negatives += 1
    if true_positives == 0:
        return {"key": "f1_score", "score": 0.0}

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return {"key": "f1_score", "score": f1_score}


evaluate(
    predict_mistral, # Your classifier
    data="Relevance Grading",
    summary_evaluators=[f1_score_summary_evaluator],
)