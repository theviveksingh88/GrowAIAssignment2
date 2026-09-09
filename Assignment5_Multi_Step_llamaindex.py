from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama


model_llama = Ollama(
    model="qwen3:0.6b",
    base_url="http://localhost:11434",
)

explanation_template = PromptTemplate(
    """
    Explain the topic "{topic}" in about 200 words.
    
    Write for a 10 years-old child
    """
)

quiz_template = PromptTemplate(
    """
    Create 5 quiz questions from this explanation

    {explanation}
    """
)

answer_template = PromptTemplate(
    """
    Generate answer key for these questions
    {quiz}
    """
)

#Running step 1

explanation = model_llama.complete(
    explanation_template.format(
        topic = "photosynthesis"
    )
)

quiz = model_llama.complete(
    quiz_template.format(
        explanation = explanation
    )
)
answer = model_llama.complete(
    answer_template.format(
        quiz = quiz
    )
)

#Print
print("Explanation")
print(explanation)
print("\n")
print("Quiz")
print(quiz)
print("\n")
print("Answer")
print(answer)

