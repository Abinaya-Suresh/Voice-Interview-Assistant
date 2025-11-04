import streamlit as st
from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
import operator
from vector_store import create_vector_stores
from audio_utils import get_audio_manager
import os
import re
import time

# Initialize audio manager
audio = get_audio_manager()

# Page configuration
st.set_page_config(
    page_title="Technical Interview System",
    page_icon="💼",
    layout="wide"
)

# Initialize LLM
@st.cache_resource
def init_llm():
    print("[TERMINAL] Initializing LLM...")
    llm = ChatOllama(model="llama3.2", temperature=0.7)
    print("[TERMINAL] LLM initialized successfully")
    return llm

llm = init_llm()

# Initialize vectorstores
@st.cache_resource
def load_vectorstores(python_path, sql_path):
    print("[TERMINAL] Loading vectorstores...")
    try:
        python_vs, sql_vs = create_vector_stores(python_path, sql_path)
        print("[TERMINAL] Vectorstores loaded successfully")
        return python_vs, sql_vs, None
    except Exception as e:
        print(f"[TERMINAL] Vectorstore loading error: {e}")
        return None, None, str(e)

# State definition
class InterviewState(TypedDict):
    current_stage: str
    user_name: str
    user_intro: str
    python_rating: int
    sql_rating: int
    questions_asked: Annotated[List[Dict], operator.add]
    current_question: str
    current_topic: str
    current_difficulty: str
    user_answer: str
    evaluation_scores: Annotated[List[Dict], operator.add]
    failed_topics: Annotated[List[str], operator.add]
    questions_count: int
    overall_score: float
    interview_complete: bool

def question_agent(state: InterviewState) -> InterviewState:
    print(f"[TERMINAL] Generating question {state['questions_count'] + 1}/10...")
    
    python_vectorstore = st.session_state.get("python_vectorstore")
    sql_vectorstore = st.session_state.get("sql_vectorstore")
    
    if state["questions_count"] >= 10:
        state["current_stage"] = "final_evaluation"
        print("[TERMINAL] All questions completed")
        return state
    
    subject = "Python" if state["questions_count"] % 2 == 0 else "SQL"
    rating = state["python_rating"] if subject == "Python" else state["sql_rating"]
    
    print(f"[TERMINAL] Subject: {subject}, Self-rating: {rating}")
    
    # Determine difficulty
    if state["evaluation_scores"]:
        recent_scores = [s["score"] for s in state["evaluation_scores"][-3:] if s["subject"] == subject]
        if recent_scores:
            avg_recent = sum(recent_scores) / len(recent_scores)
            if avg_recent < 0.4:
                difficulty = "easy"
            elif avg_recent < 0.7:
                difficulty = "intermediate"
            else:
                difficulty = "advanced"
            print(f"[TERMINAL] Difficulty adjusted based on performance: {difficulty}")
        else:
            if rating >= 8:
                difficulty = "advanced"
            elif rating >= 5:
                difficulty = "intermediate"
            else:
                difficulty = "easy"
    else:
        if rating >= 8:
            difficulty = "advanced"
        elif rating >= 5:
            difficulty = "intermediate"
        else:
            difficulty = "easy"
    
    print(f"[TERMINAL] Question difficulty: {difficulty}")
    
    failed_topics_str = ", ".join(state["failed_topics"]) if state["failed_topics"] else "none"
    
    # Get context from vectorstore
    context = ""
    try:
        print("[TERMINAL] Searching vectorstore for context...")
        if subject == "Python" and python_vectorstore:
            search_query = f"{difficulty} level Python programming concepts"
            docs = python_vectorstore.similarity_search(search_query, k=2)
            context = "\n".join([doc.page_content[:600] for doc in docs])
        elif subject == "SQL" and sql_vectorstore:
            search_query = f"{difficulty} level SQL database concepts"
            docs = sql_vectorstore.similarity_search(search_query, k=2)
            context = "\n".join([doc.page_content[:600] for doc in docs])
        print("[TERMINAL] Context retrieved from vectorstore")
    except Exception as e:
        print(f"[TERMINAL] Context retrieval error: {e}")
        context = ""
    
    # Generate question
    if context:
        question_prompt = f"""You are an interviewer. Generate ONE clear {difficulty} level {subject} interview question based on this reference material.

Reference Material:
{context[:1000]}

Requirements:
- Difficulty: {difficulty}
- Subject: {subject}
- Topics to avoid: {failed_topics_str}
- Make it practical and specific
- Output ONLY the question text, nothing else"""
    else:
        question_prompt = f"""You are a technical interviewer. Generate ONE clear {difficulty} level {subject} interview question.

Requirements:
- Difficulty: {difficulty}
- Subject: {subject}
- Topics to avoid: {failed_topics_str}
- Make it practical and concise
- Output ONLY the question text"""
    
    try:
        print("[TERMINAL] Calling LLM to generate question...")
        response = llm.invoke(question_prompt)
        question = response.content.strip()
        print("[TERMINAL] Question generated successfully")
    except Exception as e:
        print(f"[TERMINAL] LLM error: {e}, using fallback question")
        fallback_questions = {
            "Python": {
                "easy": "What is the difference between a list and a tuple in Python?",
                "intermediate": "Explain how decorators work in Python with an example.",
                "advanced": "How does Python's Global Interpreter Lock (GIL) affect multi-threading?"
            },
            "SQL": {
                "easy": "What is the difference between WHERE and HAVING clauses?",
                "intermediate": "Explain the different types of SQL joins with examples.",
                "advanced": "How would you optimize a slow-running SQL query?"
            }
        }
        question = fallback_questions.get(subject, {}).get(difficulty, "What do you know about this topic?")
    
    state["current_question"] = question
    state["current_topic"] = subject
    state["current_difficulty"] = difficulty
    state["questions_count"] += 1
    
    # Stay in questioning stage - don't advance until answer is submitted
    state["current_stage"] = "awaiting_answer"
    
    return state

def evaluation_agent(state: InterviewState) -> InterviewState:
    print(f"[TERMINAL] Evaluating answer for question {state['questions_count']}...")
    
    python_vectorstore = st.session_state.get("python_vectorstore")
    sql_vectorstore = st.session_state.get("sql_vectorstore")
    
    # Get context for evaluation
    context = ""
    try:
        print("[TERMINAL] Retrieving evaluation context...")
        if state["current_topic"] == "Python" and python_vectorstore:
            docs = python_vectorstore.similarity_search(state["current_question"], k=2)
            context = "\n".join([doc.page_content[:500] for doc in docs])
        elif state["current_topic"] == "SQL" and sql_vectorstore:
            docs = sql_vectorstore.similarity_search(state["current_question"], k=2)
            context = "\n".join([doc.page_content[:500] for doc in docs])
        print("[TERMINAL] Evaluation context retrieved")
    except Exception as e:
        print(f"[TERMINAL] Context retrieval error: {e}")
        context = ""
    
    if context:
        eval_prompt = f"""Evaluate this technical interview answer using the reference material.

Reference Material:
{context[:1000]}

Question: {state['current_question']}
Candidate's Answer: {state['user_answer']}

Provide your evaluation in EXACTLY this format:
SCORE: [a number between 0.0 and 1.0, like 0.7]
FEEDBACK: [2-3 sentences explaining the score]"""
    else:
        eval_prompt = f"""Evaluate this technical interview answer.

Question: {state['current_question']}
Subject: {state['current_topic']}
Difficulty: {state['current_difficulty']}
Candidate's Answer: {state['user_answer']}

Provide your evaluation in EXACTLY this format:
SCORE: [a number between 0.0 and 1.0, like 0.7]
FEEDBACK: [2-3 sentences explaining the score]"""
    
    try:
        print("[TERMINAL] Calling LLM for evaluation...")
        response = llm.invoke(eval_prompt)
        response_text = response.content
        
        # Parse score
        try:
            lines = response_text.split('\n')
            score_line = [line for line in lines if 'SCORE' in line.upper()][0]
            score_text = score_line.split(':')[1].strip()
            numbers = re.findall(r'0\.\d+|\d+\.\d+|\d+', score_text)
            if numbers:
                score = float(numbers[0])
                if score > 1:
                    score = score / 100
                score = max(0.0, min(1.0, score))
            else:
                score = 0.5
        except:
            response_lower = response_text.lower()
            if any(word in response_lower for word in ['excellent', 'perfect', 'correct', 'great', 'outstanding']):
                score = 0.9
            elif any(word in response_lower for word in ['good', 'solid', 'decent', 'adequate']):
                score = 0.7
            elif any(word in response_lower for word in ['partial', 'some', 'fair', 'okay']):
                score = 0.5
            elif any(word in response_lower for word in ['poor', 'incorrect', 'wrong', 'lacking']):
                score = 0.3
            else:
                score = 0.5
        
        # Extract feedback
        feedback_start = response_text.upper().find('FEEDBACK:')
        if feedback_start != -1:
            feedback = response_text[feedback_start + 9:].strip()
        else:
            feedback = response_text.strip()
        
        feedback = feedback[:300] if len(feedback) > 300 else feedback
        
        print(f"[TERMINAL] Evaluation complete - Score: {score*100:.0f}/100")
        
    except Exception as e:
        print(f"[TERMINAL] Evaluation error: {e}")
        score = 0.5
        feedback = "Answer received. Unable to provide detailed feedback."
    
    # Store evaluation
    state["evaluation_scores"].append({
        "question_num": state["questions_count"],
        "subject": state["current_topic"],
        "difficulty": state["current_difficulty"],
        "score": score,
        "feedback": feedback,
        "question": state["current_question"],
        "answer": state["user_answer"]
    })
    
    # Track failed topics
    if score < 0.4:
        topic_key = f"{state['current_topic']}_{state['current_difficulty']}"
        if topic_key not in state["failed_topics"]:
            state["failed_topics"].append(topic_key)
            print(f"[TERMINAL] Added to failed topics: {topic_key}")
    
    # Continue to next question or finish
    if state["questions_count"] < 10:
        state["current_stage"] = "questioning"
    else:
        state["current_stage"] = "final_evaluation"
    
    return state

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.stage = "setup"
    st.session_state.interview_state = None
    st.session_state.voice_input = ""
    st.session_state.question_start_time = None

# Check if running with streamlit
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        print("\n" + "="*60)
        print("ERROR: This application must be run with Streamlit!")
        print("="*60)
        print("\nPlease run using:")
        print("  streamlit run app_with_audio.py")
        print("="*60)
        exit(1)
except:
    pass

# Main UI
st.title("🎙️ Voice-Based Technical Interview System")

# Setup stage - Auto-load with default paths
if st.session_state.stage == "setup":
    # Default paths
    python_path = r"C:\Users\abina\Desktop\Resume Tracker\Interview_prep\python_book.pdf"
    sql_path = r"C:\Users\abina\Desktop\Resume Tracker\Interview_prep\sql_book.pdf"
    
    with st.spinner("🔄 Initializing interview system..."):
        python_vs, sql_vs, error = load_vectorstores(python_path, sql_path)
        
        if error:
            st.warning("⚠️ Continuing without vectorstore context...")
        
        st.session_state.python_vectorstore = python_vs
        st.session_state.sql_vectorstore = sql_vs
        st.session_state.stage = "greeting"
        st.rerun()

# Greeting stage
elif st.session_state.stage == "greeting":
    st.header("🎤 Welcome to the Technical Interview")
    
    greeting_text = "Hello! Welcome to the technical interview. May I know your name?"
    
    # Auto-speak greeting on first load - BLOCKING
    if "greeting_spoken" not in st.session_state:
        audio.speak(greeting_text)  # Blocking - wait for speech to finish
        st.session_state.greeting_spoken = True
        st.session_state.greeting_start_time = time.time()
        st.rerun()
    
    st.write(greeting_text)
    
    # Calculate remaining time (30 seconds)
    elapsed_time = time.time() - st.session_state.greeting_start_time
    remaining_time = max(0, 30 - int(elapsed_time))
    
    st.info(f"⏱️ Time remaining: {remaining_time} seconds")
    
    # Auto-skip after 30 seconds
    if remaining_time == 0:
        skip_msg = "No response received. Moving to next step."
        audio.speak(skip_msg)
        st.warning(f"⏰ {skip_msg}")
        st.session_state.user_name_input = "Guest"
        del st.session_state["greeting_spoken"]
        st.session_state.stage = "begin_interview"
        time.sleep(1)
        st.rerun()
    
    st.info("👉 Click 'Start Recording' button, wait for the ready message, then speak your name clearly")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎤 Start Recording", key="record_name", type="primary"):
            st.info("🎤 Listening... Speak your name NOW")
            voice_text = audio.listen(timeout=15, phrase_time_limit=15)
            if voice_text:
                st.session_state.user_name_input = voice_text
                st.success(f"✓ Name recorded: {voice_text}")
                del st.session_state["greeting_spoken"]
                st.session_state.stage = "begin_interview"
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ No speech detected. Please check your microphone and try again.")
    
    with col2:
        if st.button("🔊 Repeat Question"):
            audio.speak(greeting_text)
    
    with col3:
        if st.button("⏭️ Skip"):
            st.session_state.user_name_input = "Guest"
            del st.session_state["greeting_spoken"]
            st.session_state.stage = "begin_interview"
            st.rerun()

# Begin interview confirmation
elif st.session_state.stage == "begin_interview":
    st.header(f"Hello {st.session_state.user_name_input}!")
    
    begin_text = "Let's begin our interview process. Are you ready?"
    
    if "begin_spoken" not in st.session_state:
        audio.speak(begin_text)  # Blocking call
        st.session_state.begin_spoken = True
        st.session_state.begin_start_time = time.time()
        st.rerun()
    
    st.write(begin_text)
    
    # Calculate remaining time (30 seconds)
    elapsed_time = time.time() - st.session_state.begin_start_time
    remaining_time = max(0, 30 - int(elapsed_time))
    
    st.info(f"⏱️ Time remaining: {remaining_time} seconds")
    
    # Auto-skip after 30 seconds
    if remaining_time == 0:
        skip_msg = "No response received. Starting interview."
        audio.speak(skip_msg)
        st.warning(f"⏰ {skip_msg}")
        del st.session_state["begin_spoken"]
        st.session_state.stage = "introduction"
        time.sleep(1)
        st.rerun()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Yes, I'm Ready"):
            del st.session_state["begin_spoken"]
            st.session_state.stage = "introduction"
            st.rerun()
    
    with col2:
        if st.button("⏭️ Skip"):
            del st.session_state["begin_spoken"]
            st.session_state.stage = "introduction"
            st.rerun()

# Introduction stage
elif st.session_state.stage == "introduction":
    st.header("Introduction")
    
    intro_text = "Please tell us about yourself. Share your background, experience, and what brings you here today. You have up to 2 minutes to respond."
    
    if "intro_spoken" not in st.session_state:
        audio.speak(intro_text)  # Blocking call - wait for speech to finish
        st.session_state.intro_spoken = True
        st.session_state.intro_start_time = time.time()
        st.rerun()
    
    st.write(intro_text)
    
    # Calculate remaining time (30 seconds)
    elapsed_time = time.time() - st.session_state.intro_start_time
    remaining_time = max(0, 30 - int(elapsed_time))
    
    st.info(f"⏱️ Time remaining: {remaining_time} seconds")
    
    # Auto-skip after 30 seconds
    if remaining_time == 0:
        skip_msg = "No response received. Moving to skills assessment."
        audio.speak(skip_msg)
        st.warning(f"⏰ {skip_msg}")
        st.session_state.user_intro = "No introduction provided"
        del st.session_state["intro_spoken"]
        st.session_state.stage = "rating"
        time.sleep(1)
        st.rerun()
    
    st.info("🎤 Click 'Start Recording' and speak your introduction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎤 Start Recording", key="record_intro"):
            with st.spinner("🎤 Listening... Speak your introduction"):
                voice_text = audio.listen(timeout=15, phrase_time_limit=120)
                if voice_text and len(voice_text.strip()) > 20:
                    st.session_state.user_intro = voice_text
                    st.success("Introduction recorded!")
                    del st.session_state["intro_spoken"]
                    st.session_state.stage = "rating"
                    st.rerun()
                elif voice_text:
                    st.error("Introduction too short. Please provide more details.")
                else:
                    st.warning("No speech detected. Please try again.")
    
    with col2:
        if st.button("🔊 Repeat Question"):
            audio.speak(intro_text)
    
    with col3:
        if st.button("⏭️ Skip"):
            st.session_state.user_intro = "No introduction provided"
            del st.session_state["intro_spoken"]
            st.session_state.stage = "rating"
            st.rerun()

# Rating stage
elif st.session_state.stage == "rating":
    st.header("Skills Assessment")
    
    rating_text = "Please rate your Python and SQL skills on a scale of 0 to 10."
    
    if "rating_spoken" not in st.session_state:
        audio.speak(rating_text)  # Blocking call
        st.session_state.rating_spoken = True
        st.session_state.rating_start_time = time.time()
        st.rerun()
    
    st.write(rating_text)
    
    # Calculate remaining time (30 seconds)
    elapsed_time = time.time() - st.session_state.rating_start_time
    remaining_time = max(0, 30 - int(elapsed_time))
    
    st.info(f"⏱️ Time remaining: {remaining_time} seconds")
    
    # Auto-skip after 30 seconds with default ratings
    if remaining_time == 0:
        skip_msg = "No response received. Starting interview with default ratings."
        audio.speak(skip_msg)
        st.warning(f"⏰ {skip_msg}")
        
        st.session_state.python_rating = 5
        st.session_state.sql_rating = 5
        
        # Initialize interview state
        st.session_state.interview_state = {
            "current_stage": "questioning",
            "user_name": st.session_state.user_name_input,
            "user_intro": st.session_state.get("user_intro", ""),
            "python_rating": 5,
            "sql_rating": 5,
            "questions_asked": [],
            "current_question": "",
            "current_topic": "",
            "current_difficulty": "",
            "user_answer": "",
            "evaluation_scores": [],
            "failed_topics": [],
            "questions_count": 0,
            "overall_score": 0.0,
            "interview_complete": False
        }
        
        del st.session_state["rating_spoken"]
        st.session_state.stage = "interview"
        time.sleep(1)
        st.rerun()
    
    col1, col2 = st.columns(2)
    
    with col1:
        python_rating = st.slider("Python Skills (0-10)", 0, 10, 5, key="python_slider")
    
    with col2:
        sql_rating = st.slider("SQL Skills (0-10)", 0, 10, 5, key="sql_slider")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Interview"):
            st.session_state.python_rating = python_rating
            st.session_state.sql_rating = sql_rating
            
            # Initialize interview state
            st.session_state.interview_state = {
                "current_stage": "questioning",
                "user_name": st.session_state.user_name_input,
                "user_intro": st.session_state.get("user_intro", ""),
                "python_rating": python_rating,
                "sql_rating": sql_rating,
                "questions_asked": [],
                "current_question": "",
                "current_topic": "",
                "current_difficulty": "",
                "user_answer": "",
                "evaluation_scores": [],
                "failed_topics": [],
                "questions_count": 0,
                "overall_score": 0.0,
                "interview_complete": False
            }
            
            del st.session_state["rating_spoken"]
            st.session_state.stage = "interview"
            st.rerun()
    
    with col2:
        if st.button("⏭️ Skip (Use Defaults)"):
            st.session_state.python_rating = 5
            st.session_state.sql_rating = 5
            
            # Initialize interview state
            st.session_state.interview_state = {
                "current_stage": "questioning",
                "user_name": st.session_state.user_name_input,
                "user_intro": st.session_state.get("user_intro", ""),
                "python_rating": 5,
                "sql_rating": 5,
                "questions_asked": [],
                "current_question": "",
                "current_topic": "",
                "current_difficulty": "",
                "user_answer": "",
                "evaluation_scores": [],
                "failed_topics": [],
                "questions_count": 0,
                "overall_score": 0.0,
                "interview_complete": False
            }
            
            del st.session_state["rating_spoken"]
            st.session_state.stage = "interview"
            st.rerun()

# Interview stage
elif st.session_state.stage == "interview":
    state = st.session_state.interview_state
    
    # Generate question if needed
    if state["current_stage"] == "questioning":
        state = question_agent(state)
        st.session_state.interview_state = state
        st.session_state.question_start_time = time.time()
        st.session_state.question_spoken = False
    
    # Display question and wait for answer
    if state["current_stage"] == "awaiting_answer":
        st.header(f"Question {state['questions_count']}/10")
        st.subheader(f"📌 {state['current_topic']} - {state['current_difficulty'].upper()}")
        
        st.write("---")
        st.write(f"**{state['current_question']}**")
        st.write("---")
        
        # Auto-speak question once - BLOCKING
        if not st.session_state.get("question_spoken", False):
            audio.speak(state["current_question"])  # Blocking call
            st.session_state.question_spoken = True
            st.rerun()
        
        # Calculate remaining time (30 seconds per question)
        elapsed_time = time.time() - st.session_state.question_start_time
        remaining_time = max(0, 30 - int(elapsed_time))
        
        st.info(f"⏱️ Time remaining: {remaining_time} seconds")
        
        # Auto-skip after 30 seconds
        if remaining_time == 0:
            skip_msg = "Time's up! Moving to next question."
            audio.speak(skip_msg)
            st.warning(f"⏰ {skip_msg}")
            state["user_answer"] = "No answer provided (timeout)"
            state["questions_asked"].append({
                "question": state["current_question"],
                "subject": state["current_topic"],
                "difficulty": state["current_difficulty"],
                "answer": "No answer provided (timeout)"
            })
            state["current_stage"] = "evaluation"
            st.session_state.interview_state = state
            time.sleep(2)
            st.rerun()
        
        # Show previous progress in sidebar
        if state["evaluation_scores"]:
            with st.sidebar:
                st.subheader("Progress")
                st.write(f"Completed: {len(state['evaluation_scores'])}/10 questions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎤 Record Answer", key=f"record_{state['questions_count']}"):
                with st.spinner("🎤 Listening... Speak your answer"):
                    voice_text = audio.listen(timeout=15, phrase_time_limit=30)
                    if voice_text and voice_text.strip():
                        state["user_answer"] = voice_text
                        state["questions_asked"].append({
                            "question": state["current_question"],
                            "subject": state["current_topic"],
                            "difficulty": state["current_difficulty"],
                            "answer": voice_text
                        })
                        state["current_stage"] = "evaluation"
                        st.session_state.interview_state = state
                        st.success("Answer recorded!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.warning("No speech detected. Please try again.")
        
        with col2:
            if st.button("⏭️ Skip Question", key=f"skip_{state['questions_count']}"):
                skip_msg = "Let's move to the next question."
                audio.speak(skip_msg)
                state["user_answer"] = "Question skipped by user"
                state["questions_asked"].append({
                    "question": state["current_question"],
                    "subject": state["current_topic"],
                    "difficulty": state["current_difficulty"],
                    "answer": "Question skipped by user"
                })
                state["current_stage"] = "evaluation"
                st.session_state.interview_state = state
                time.sleep(1)
                st.rerun()
        
        with col3:
            if st.button("🔊 Repeat Question", key=f"repeat_{state['questions_count']}"):
                audio.speak(state["current_question"])
    
    # Evaluate the answer silently
    elif state["current_stage"] == "evaluation":
        with st.spinner("Evaluating your answer..."):
            state = evaluation_agent(state)
            st.session_state.interview_state = state
        
        # Move to next question or final results without showing individual scores
        if state["current_stage"] == "questioning":
            st.rerun()
        else:
            st.session_state.stage = "results"
            st.rerun()
    
    # Final evaluation
    elif state["current_stage"] == "final_evaluation":
        print("[TERMINAL] Generating final report...")
        if state["evaluation_scores"]:
            total_score = sum([s["score"] for s in state["evaluation_scores"]])
            state["overall_score"] = (total_score / len(state["evaluation_scores"])) * 100
        
        state["interview_complete"] = True
        st.session_state.interview_state = state
        st.session_state.stage = "results"
        st.rerun()

# Results stage
elif st.session_state.stage == "results":
    state = st.session_state.interview_state
    
    if state["evaluation_scores"]:
        total_score = sum([s["score"] for s in state["evaluation_scores"]])
        overall_score = (total_score / len(state["evaluation_scores"])) * 100
        
        st.header("🎉 Interview Complete - Final Results")
        
        # Overall score display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Score", f"{overall_score:.1f}/100")
        
        python_scores = [s["score"] for s in state["evaluation_scores"] if s["subject"] == "Python"]
        sql_scores = [s["score"] for s in state["evaluation_scores"] if s["subject"] == "SQL"]
        
        if python_scores:
            with col2:
                python_avg = (sum(python_scores)/len(python_scores))*100
                st.metric("Python Average", f"{python_avg:.1f}/100")
        
        if sql_scores:
            with col3:
                sql_avg = (sum(sql_scores)/len(sql_scores))*100
                st.metric("SQL Average", f"{sql_avg:.1f}/100")
        
        # Performance summary
        if overall_score >= 80:
            grade = "A"
            summary = "Excellent performance! Strong technical skills demonstrated."
        elif overall_score >= 70:
            grade = "B"
            summary = "Good performance. Solid understanding with minor gaps."
        elif overall_score >= 60:
            grade = "C"
            summary = "Fair performance. Some areas need strengthening."
        elif overall_score >= 50:
            grade = "D"
            summary = "Below average. Focus on building stronger foundations."
        else:
            grade = "F"
            summary = "Needs significant improvement. Review fundamental concepts."
        
        st.subheader(f"📊 Final Grade: {grade}")
        st.write(summary)
        
        # Thank you message
        thankyou_text = f"Thank you {state['user_name']} for completing the interview. Your overall score is {overall_score:.0f} out of 100. Your final grade is {grade}. {summary}"
        
        # Auto-speak results once - BLOCKING
        if "results_spoken" not in st.session_state:
            audio.speak(thankyou_text)
            st.session_state.results_spoken = True
            st.rerun()
        
        st.success("✅ Thank you for participating in the interview!")
        
        # Difficulty distribution
        st.subheader("📈 Question Difficulty Distribution")
        easy_count = len([s for s in state["evaluation_scores"] if s["difficulty"] == "easy"])
        inter_count = len([s for s in state["evaluation_scores"] if s["difficulty"] == "intermediate"])
        adv_count = len([s for s in state["evaluation_scores"] if s["difficulty"] == "advanced"])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Easy", easy_count)
        col2.metric("Intermediate", inter_count)
        col3.metric("Advanced", adv_count)
        
        # Score summary only (no feedback)
        st.subheader("📋 Question Scores Summary")
        for s in state["evaluation_scores"]:
            st.write(f"**Q{s['question_num']}:** {s['subject']} ({s['difficulty']}) - **{s['score']*100:.0f}/100**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔊 Hear Results Again"):
                audio.speak(thankyou_text)
        
        with col2:
            if st.button("🔄 Start New Interview"):
                print("[TERMINAL] Starting new interview...")
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

print("[TERMINAL] App is running and ready for user interaction")