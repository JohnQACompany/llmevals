from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


input_text = "Wat is de hoofdstad van Frankrijk en antwoord in het Engels?"
expected_output = "The capital of France is Paris."

# Plak hier zoveel gevonden antwoorden als je wilt, bijvoorbeeld 10
actual_outputs = [
    "Paris is the capital of France.",
    "The capital of France is Paris.",
    "France's capital is Paris.",
    "Paris.",
    "The capital city of France is Paris.",
    "Paris is the capital of France. I hope this helps.",
    "France has Paris as its capital.",
    "The answer is Paris, which is the capital of France.",
    "Paris is the capital of France, and I am answering in English as requested.",
    "Lyon is the capital of France."
]

metric = GEval(
    name="Expected Answer Alignment",
    criteria=(
        "Beoordeel in hoeverre het daadwerkelijke antwoord inhoudelijk overeenkomt "
        "met het verwachte antwoord. Kijk naar feitelijke juistheid, semantische overeenkomst, "
        "volledigheid en het volgen van expliciete instructies uit de input, zoals gevraagde taal. "
        "Kleine verschillen in formulering zijn toegestaan zolang de betekenis gelijk blijft."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.8,
)

test_cases = [
    LLMTestCase(
        input=input_text,
        expected_output=expected_output,
        actual_output=actual_output.strip(),
    )
    for actual_output in actual_outputs
]

results = evaluate(test_cases, [metric])
