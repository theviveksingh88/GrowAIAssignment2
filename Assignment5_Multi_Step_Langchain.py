from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(
    model= "qwen3:0.6b",
    base_url="http://localhost:11434"
)

#LangChain Pipeline
explanation_prompt = ChatPromptTemplate.from_template("""
Explain the topic "{topic}" in about 200 words.

Write for a 10-year-old child.
Use simple English.
""")

quiz_prompt = ChatPromptTemplate.from_template("""
You are a teacher.

Based on the explanation below, create exactly 5 quiz questions.

Explanation:

{explanation}
""")

answer_prompt = ChatPromptTemplate.from_template("""
Below are quiz questions.

Generate an answer key.

Quiz:

{quiz}
""")

parser = StrOutputParser()

explanation_chain = (
    explanation_prompt
    | model
    | parser
)

quiz_chain = (
    quiz_prompt
    | model
    | parser
)

answer_chain = (
    answer_prompt
    | model
    | parser
)

topic = "photosynthesis"

#Step 1

explanation = explanation_chain.invoke({
    "topic": topic
})

#Step 2

quiz = quiz_chain.invoke({
    "explanation": explanation
})

#Step 3

answer = answer_chain.invoke({
    "quiz": quiz
})

print("Explanation:")
print(explanation)
print("\n")
print("Quiz:")
print(quiz)
print("\n")
print("Answer:")
print(answer)


