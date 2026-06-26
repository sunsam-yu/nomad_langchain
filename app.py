import random

import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

QUIZ_FUNCTION = {
    "type": "function",
    "function": {
        "name": "create_quiz",
        "description": "Create a multiple-choice quiz from the given context.",
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "description": "A list of quiz questions.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The quiz question.",
                            },
                            "explanation": {
                                "type": "string",
                                "description": "A 2-3 sentence explanation of why the correct answer is correct.",
                            },
                            "answers": {
                                "type": "array",
                                "description": "Exactly four possible answers.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "answer": {
                                            "type": "string",
                                            "description": "A possible answer.",
                                        },
                                        "correct": {
                                            "type": "boolean",
                                            "description": "Whether this answer is correct.",
                                        },
                                    },
                                    "required": ["answer", "correct"],
                                },
                            },
                        },
                        "required": ["question", "explanation", "answers"],
                    },
                }
            },
            "required": ["questions"],
        },
    },
}

# 위의 QUIZ_FUNCTION은 실제 Python 함수가 아니라,
# LLM이 아래와 같은 구조로 답하도록 강제하는 "출력 형식 설계도"입니다.
#
# {
#     "questions": [
#         {
#             "question": "What is Python?",
#             "explanation": "Python is a programming language...",
#             "answers": [
#                 {"answer": "A programming language", "correct": True},
#                 {"answer": "A snake only", "correct": False},
#                 {"answer": "A database", "correct": False},
#                 {"answer": "An operating system", "correct": False},
#             ],
#         }
#     ]
# }


def reset_quiz():
    st.session_state["quiz"] = None
    st.session_state["submitted"] = False
    st.session_state["results"] = None
    st.session_state["attempt"] = st.session_state.get("attempt", 0) + 1
    st.session_state["balloons_shown"] = False


@st.cache_data(show_spinner="Searching Wikipedia...")
def search_wikipedia(topic):
    retriever = WikipediaRetriever(
        top_k_results=3,
        lang="en",
    )
    docs = retriever.invoke(topic)
    return "\n\n".join(doc.page_content for doc in docs)


def shuffle_answers(quiz):
    for question in quiz["questions"]:
        random.shuffle(question["answers"])
    return quiz


def generate_quiz(topic, difficulty, openai_api_key):
    context = search_wikipedia(topic)

    if not context:
        raise ValueError("No Wikipedia content found for this topic.")

    context = context[:8000]

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        openai_api_key=openai_api_key,
    )

    llm_with_tools = llm.bind_tools(
        [QUIZ_FUNCTION],
        tool_choice={
            "type": "function",
            "function": {
                "name": "create_quiz",
            },
        },
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a quiz generator.

Create a multiple-choice quiz using ONLY the provided Wikipedia context.

Rules:
- Generate exactly 5 questions.
- Each question must have exactly 4 answers.
- Each question must have exactly 1 correct answer.
- For each question, include a 2-3 sentence explanation of why the correct answer is correct.
- The answers must be shuffled.
- Do not write explanations outside the function call.
- Do not use information outside the context.

Difficulty rules:
- Easy questions should ask about clear, general, easy-to-find facts.
- Hard questions should require more detailed understanding and use more confusing distractors.
                """,
            ),
            (
                "human",
                """
Topic: {topic}
Difficulty: {difficulty}

Wikipedia context:
{context}
                """,
            ),
        ]
    )

    chain = prompt | llm_with_tools
    response = chain.invoke(
        {
            "topic": topic,
            "difficulty": difficulty,
            "context": context,
        }
    )

    if not response.tool_calls:
        raise ValueError(
         "The model did not return a valid quiz. Please try again with a clearer Wikipedia topic."
        )

    quiz = response.tool_calls[0]["args"]

    if not quiz.get("questions"):
        raise ValueError(
            "The quiz was empty. Please try again with a more specific Wikipedia topic."
        )

    return shuffle_answers(quiz)


def grade_quiz(quiz):
    results = []
    score = 0

    for index, question in enumerate(quiz["questions"]):
        key = f"question_{index}_{st.session_state['attempt']}"
        selected_answer = st.session_state.get(key)

        correct_answer = None

        for answer in question["answers"]:
            if answer["correct"]:
                correct_answer = answer["answer"]
                break

        is_correct = selected_answer == correct_answer

        if is_correct:
            score += 1

        results.append(
            {
                "question": question["question"],
                "selected_answer": selected_answer,
                "correct_answer": correct_answer,
                "explanation": question.get(
                    "explanation",
                    "No explanation was provided.",
                ),
                "is_correct": is_correct,
            }
        )

    return score, results


if "quiz" not in st.session_state:
    st.session_state["quiz"] = None

if "submitted" not in st.session_state:
    st.session_state["submitted"] = False

if "results" not in st.session_state:
    st.session_state["results"] = None

if "attempt" not in st.session_state:
    st.session_state["attempt"] = 0

if "balloons_shown" not in st.session_state:
    st.session_state["balloons_shown"] = False


with st.sidebar:
    st.header("Settings")

    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
    )

    difficulty = st.selectbox(
        "Difficulty",
        (
            "Easy",
            "Hard",
        ),
    )

    st.markdown("---")
    st.markdown(
        "[GitHub Repository](https://github.com/sunsam-yu/nomad_langchain)"
    )


st.title("❓ QuizGPT")

st.markdown(
    """
Search a Wikipedia topic, generate a quiz, and test your knowledge.
    """
)

topic = st.text_input(
    "Wikipedia topic",
    placeholder="For example: Python programming language, Rome, Artificial intelligence",
)

if st.button("Generate quiz"):
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
    elif not topic:
        st.warning("Please enter a Wikipedia topic.")
    else:
        reset_quiz()

        with st.spinner("Generating quiz..."):
            try:
                quiz = generate_quiz(
                    topic,
                    difficulty,
                    openai_api_key,
                )
                st.session_state["quiz"] = quiz
                st.success("Quiz generated!")
            except Exception as e:
                st.error(f"Error: {e}")


quiz = st.session_state["quiz"]

if quiz:
    st.markdown("## Quiz")

    for index, question in enumerate(quiz["questions"]):
        st.markdown(f"### Question {index + 1}")
        st.markdown(question["question"])

        answer_options = [
            answer["answer"]
            for answer in question["answers"]
        ]

        st.radio(
            "Choose one answer:",
            answer_options,
            index=None,
            key=f"question_{index}_{st.session_state['attempt']}",
            disabled=st.session_state["submitted"],
        )

    if not st.session_state["submitted"]:
        if st.button("Submit answers"):
            score, results = grade_quiz(quiz)

            st.session_state["submitted"] = True
            st.session_state["results"] = {
                "score": score,
                "items": results,
            }

            st.rerun()

    if st.session_state["submitted"]:
        total = len(quiz["questions"])
        score = st.session_state["results"]["score"]

        st.markdown("## Result")
        st.subheader(f"Score: {score} / {total}")

        for result in st.session_state["results"]["items"]:
            if result["is_correct"]:
                st.success(f"Correct: {result['question']}")
            else:
                st.error(f"Wrong: {result['question']}")

                if result["selected_answer"] is None:
                    st.write("Your answer: Not selected")
                else:
                    st.write(f"Your answer: {result['selected_answer']}")

                st.write(f"Correct answer: {result['correct_answer']}")
                st.info(f"Explanation: {result['explanation']}")

        if score == total:
            st.success("Perfect score!")

            if not st.session_state["balloons_shown"]:
                st.balloons()
                st.session_state["balloons_shown"] = True
        else:
            st.warning("You did not get a perfect score. Try again!")

            if st.button("Retake test"):
                st.session_state["submitted"] = False
                st.session_state["results"] = None
                st.session_state["attempt"] += 1
                st.session_state["balloons_shown"] = False
                st.rerun()
else:
    st.info("Enter a topic and click 'Generate quiz' to start.")