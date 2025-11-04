"""
Voice-Based Technical Interview System - Version 2
Enhanced with strict question uniqueness and better distribution
"""

from typing import TypedDict, Annotated, List, Dict
from langchain_ollama import ChatOllama
import operator
from vector_store import create_vector_stores
from audio_utils import get_audio_manager
import time
import re
import sys
import io
import random

# Suppress terminal output from audio_utils
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

# Initialize audio manager without terminal prints
with HiddenPrints():
    audio = get_audio_manager()

# Initialize LLM
def init_llm():
    return ChatOllama(model="llama3.2", temperature=0.7)

llm = init_llm()

# Load vectorstores
def load_vectorstores(python_path, sql_path):
    try:
        python_vs, sql_vs = create_vector_stores(python_path, sql_path)
        return python_vs, sql_vs
    except:
        return None, None

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
    asked_question_texts: Annotated[List[str], operator.add]  # Track exact questions

def get_fallback_questions():
    """Comprehensive question bank with unique questions"""
    return {
        "Python": {
            "easy": [
                "What is the difference between a list and a tuple in Python?",
                "What does the 'self' keyword represent in Python classes?",
                "What is the purpose of the 'pass' statement in Python?",
                "What is the difference between '==' and 'is' operators in Python?",
                "What are Python's built-in data types?",
                "What is the difference between 'break' and 'continue' statements?",
                "What is a Python module?",
                "What is the purpose of '__init__' method in Python classes?",
                "What is the difference between local and global variables?",
                "What are Python docstrings and why are they used?"
            ],
            "intermediate": [
                "Explain what decorators are in Python and when you would use them.",
                "What is the difference between shallow copy and deep copy in Python?",
                "Explain how exception handling works in Python using try-except blocks.",
                "What are lambda functions in Python and when should you use them?",
                "What is the difference between 'append()' and 'extend()' methods in lists?",
                "Explain the concept of list comprehensions in Python.",
                "What are generators in Python and how do they differ from regular functions?",
                "What is the purpose of the '__name__' variable in Python?",
                "Explain what Python's 'with' statement does and when to use it.",
                "What is the difference between instance methods and class methods?"
            ],
            "advanced": [
                "What is the Global Interpreter Lock (GIL) and how does it affect Python threading?",
                "Explain how Python's garbage collection mechanism works.",
                "What are metaclasses in Python and what problems do they solve?",
                "Explain the concept of context managers in Python and how they work.",
                "What is the difference between threads and processes in Python?",
                "Explain Python's method resolution order (MRO) in multiple inheritance.",
                "What are Python descriptors and how do they work?",
                "Explain the concept of closures in Python.",
                "What is the difference between 'new' and 'init' methods in Python?",
                "How does Python's memory management differ from other languages?"
            ]
        },
        "SQL": {
            "easy": [
                "What is the difference between WHERE and HAVING clauses in SQL?",
                "What is a primary key in a database table?",
                "What does the DISTINCT keyword do in SQL?",
                "What is the difference between DELETE and TRUNCATE commands?",
                "What is a foreign key in SQL?",
                "What does the ORDER BY clause do in SQL?",
                "What is the difference between CHAR and VARCHAR data types?",
                "What is a NULL value in SQL?",
                "What does the COUNT function do in SQL?",
                "What is the purpose of the GROUP BY clause?"
            ],
            "intermediate": [
                "Explain the different types of SQL joins and when to use each one.",
                "What is database normalization and why is it important?",
                "Explain what a database index is and how it improves performance.",
                "What is the difference between a view and a table in SQL?",
                "What are subqueries and when would you use them?",
                "Explain the difference between UNION and UNION ALL.",
                "What is a stored procedure and what are its benefits?",
                "What are SQL triggers and when are they used?",
                "Explain what a database transaction is.",
                "What is the difference between primary and unique constraints?"
            ],
            "advanced": [
                "Explain the ACID properties in database transactions.",
                "What is the difference between clustered and non-clustered indexes?",
                "Explain what database deadlocks are and how to prevent them.",
                "What are the different isolation levels in database transactions?",
                "Explain the concept of database sharding and when to use it.",
                "What is query optimization and how do database engines perform it?",
                "Explain the difference between pessimistic and optimistic locking.",
                "What are materialized views and how do they differ from regular views?",
                "Explain the concept of database partitioning.",
                "What is the CAP theorem in distributed databases?"
            ]
        }
    }

def is_question_duplicate(question, asked_questions, threshold=0.6):
    """
    Check if a question is too similar to previously asked questions
    Returns True if duplicate, False if unique
    """
    question_lower = question.lower()
    question_words = set(question_lower.split())
    
    for asked in asked_questions:
        asked_lower = asked.lower()
        asked_words = set(asked_lower.split())
        
        # Calculate similarity based on word overlap
        if len(question_words) > 0:
            overlap = question_words.intersection(asked_words)
            similarity = len(overlap) / len(question_words)
            
            if similarity > threshold:
                return True
    
    return False

def question_agent(state: InterviewState, python_vs, sql_vs) -> InterviewState:
    if state["questions_count"] >= 10:
        state["current_stage"] = "final_evaluation"
        return state
    
    # Calculate question distribution based on ratings
    python_rating = state["python_rating"]
    sql_rating = state["sql_rating"]
    total_rating = python_rating + sql_rating
    
    # Determine subject based on rating proportion and ensure balanced distribution
    if total_rating > 0:
        python_percentage = python_rating / total_rating
        python_questions_done = len([s for s in state["evaluation_scores"] if s["subject"] == "Python"])
        sql_questions_done = len([s for s in state["evaluation_scores"] if s["subject"] == "SQL"])
        
        # Target distribution
        target_python = int(10 * python_percentage)
        target_sql = 10 - target_python
        
        # Choose subject to maintain balance
        if python_questions_done < target_python and python_questions_done < 10:
            subject = "Python"
        elif sql_questions_done < target_sql and sql_questions_done < 10:
            subject = "SQL"
        else:
            # Fallback to rating-based selection
            subject = "Python" if python_rating >= sql_rating else "SQL"
    else:
        subject = "Python" if state["questions_count"] % 2 == 0 else "SQL"
    
    rating = state["python_rating"] if subject == "Python" else state["sql_rating"]
    
    # Store initial difficulty based on user's declared skill level for this subject
    initial_difficulty_by_rating = None
    if rating >= 8:
        initial_difficulty_by_rating = "advanced"
    elif rating >= 5:
        initial_difficulty_by_rating = "intermediate"
    else:
        initial_difficulty_by_rating = "easy"
    
    # Determine difficulty based on recent performance OR initial rating
    if state["evaluation_scores"]:
        subject_scores = [s for s in state["evaluation_scores"] if s["subject"] == subject]
        
        if subject_scores:
            # Convert last base_score (0.0-1.0) to 0-10 scale for difficulty decision
            last_score_10_scale = subject_scores[-1]["base_score"] * 10
            
            # NEW: Score-based difficulty adjustment (0-3→easy, 3-6→intermediate, 6-10→advanced)
            if last_score_10_scale < 3:
                difficulty = "easy"
            elif last_score_10_scale < 6:
                difficulty = "intermediate"
            else:
                difficulty = "advanced"
        else:
            # First question for this subject - use rating-based difficulty
            difficulty = initial_difficulty_by_rating
    else:
        # Very first question - use rating-based difficulty
        difficulty = initial_difficulty_by_rating
    
    # Get asked questions for duplicate checking
    asked_question_texts = state.get("asked_question_texts", [])
    
    # Try to generate unique question from LLM
    question = None
    max_attempts = 5
    
    for attempt in range(max_attempts):
        # Get context from vectorstore
        context = ""
        try:
            if subject == "Python" and python_vs:
                search_query = f"{difficulty} level Python programming concepts theory"
                docs = python_vs.similarity_search(search_query, k=2)
                context = "\n".join([doc.page_content[:600] for doc in docs])
            elif subject == "SQL" and sql_vs:
                search_query = f"{difficulty} level SQL database concepts theory"
                docs = sql_vs.similarity_search(search_query, k=2)
                context = "\n".join([doc.page_content[:600] for doc in docs])
        except:
            context = ""
        
        if context:
            question_prompt = f"""Generate a UNIQUE technical interview question.

Subject: {subject}
Difficulty: {difficulty}
Question Count: {state['questions_count'] + 1}/10

PREVIOUSLY ASKED QUESTIONS (DO NOT REPEAT OR REPHRASE):
{chr(10).join(asked_question_texts[-5:]) if asked_question_texts else "None"}

Reference Material:
{context[:800]}

CRITICAL REQUIREMENTS:
1. Generate a COMPLETELY DIFFERENT question from all previous ones
2. Ask ONLY about theory/concepts - NO coding, NO "write a program/query"
3. Focus on: definitions, differences, advantages, use cases, working principles
4. Keep under 30 words
5. Make it specific and clear
6. For {difficulty} level

Examples of GOOD questions:
- "What is the difference between X and Y?"
- "Explain how X works in {subject}."
- "What are the advantages of using X?"
- "When would you use X over Y?"

Output ONLY the question text:"""
            
            try:
                response = llm.invoke(question_prompt)
                candidate_question = response.content.strip().replace('"', '').replace("'", "")
                
                # Check for coding keywords
                coding_keywords = ['write a', 'write the', 'implement', 'create a function', 
                                 'write code', 'code for', 'program to', 'query to']
                has_coding = any(keyword in candidate_question.lower() for keyword in coding_keywords)
                
                # Check for duplication
                is_duplicate = is_question_duplicate(candidate_question, asked_question_texts, threshold=0.5)
                
                if not has_coding and not is_duplicate:
                    question = candidate_question
                    break
            except:
                pass
    
    # Fallback to curated question bank
    if not question:
        fallback_bank = get_fallback_questions()
        available_questions = [
            q for q in fallback_bank[subject][difficulty]
            if not is_question_duplicate(q, asked_question_texts, threshold=0.5)
        ]
        
        if available_questions:
            question = random.choice(available_questions)
        else:
            # All questions used - pick least similar
            all_questions = fallback_bank[subject][difficulty]
            question = random.choice(all_questions)
    
    # Update state
    state["current_question"] = question
    state["current_topic"] = subject
    state["current_difficulty"] = difficulty
    state["questions_count"] += 1
    state["current_stage"] = "awaiting_answer"
    state["asked_question_texts"].append(question)
    
    return state

def evaluation_agent(state: InterviewState, python_vs, sql_vs) -> InterviewState:
    # Get context for evaluation
    context = ""
    try:
        if state["current_topic"] == "Python" and python_vs:
            docs = python_vs.similarity_search(state["current_question"], k=2)
            context = "\n".join([doc.page_content[:500] for doc in docs])
        elif state["current_topic"] == "SQL" and sql_vs:
            docs = sql_vs.similarity_search(state["current_question"], k=2)
            context = "\n".join([doc.page_content[:500] for doc in docs])
    except:
        context = ""
    
    # Difficulty multipliers
    difficulty_weights = {
        "easy": 1.0,
        "intermediate": 1.5,
        "advanced": 2.0
    }
    
    difficulty = state["current_difficulty"]
    weight = difficulty_weights.get(difficulty, 1.0)
    
    eval_prompt = f"""Evaluate this technical interview answer.

Question: {state['current_question']}
Subject: {state['current_topic']}
Difficulty: {difficulty}
Candidate's Answer: {state['user_answer']}

{"Reference Material: " + context[:800] if context else ""}

SCORING GUIDELINES for {difficulty} level:
- Easy (0.0-1.0): Basic understanding and correct terminology
- Intermediate (0.0-1.0): Good explanation with context and examples
- Advanced (0.0-1.0): Expert-level depth, accuracy, and nuanced understanding

SPECIAL CASES:
- "No answer provided" or timeout → 0.0
- Completely wrong or irrelevant → 0.0-0.2
- Partial understanding → 0.3-0.5
- Good understanding → 0.6-0.8
- Excellent, complete answer → 0.9-1.0

Provide evaluation in EXACTLY this format:
SCORE: [number 0.0-1.0]
FEEDBACK: [2-3 sentences]"""
    
    try:
        response = llm.invoke(eval_prompt)
        response_text = response.content
        
        # Parse score
        try:
            lines = response_text.split('\n')
            score_line = [line for line in lines if 'SCORE' in line.upper()][0]
            score_text = score_line.split(':')[1].strip()
            numbers = re.findall(r'0\.\d+|\d+\.\d+|\d+', score_text)
            if numbers:
                base_score = float(numbers[0])
                if base_score > 1:
                    base_score = base_score / 100
                base_score = max(0.0, min(1.0, base_score))
            else:
                base_score = 0.5
        except:
            # Fallback sentiment analysis
            response_lower = response_text.lower()
            if any(word in response_lower for word in ['excellent', 'perfect', 'outstanding']):
                base_score = 0.9
            elif any(word in response_lower for word in ['good', 'solid', 'correct']):
                base_score = 0.7
            elif any(word in response_lower for word in ['partial', 'fair', 'okay']):
                base_score = 0.5
            elif any(word in response_lower for word in ['poor', 'incorrect', 'wrong']):
                base_score = 0.2
            else:
                base_score = 0.5
        
        # Special case: No answer or timeout
        if "no answer provided" in state["user_answer"].lower() or "timeout" in state["user_answer"].lower():
            base_score = 0.0
        
        weighted_score = base_score * weight
        
        # Extract feedback
        feedback_start = response_text.upper().find('FEEDBACK:')
        if feedback_start != -1:
            feedback = response_text[feedback_start + 9:].strip()
        else:
            feedback = response_text.strip()
        
        feedback = feedback[:300]
        
    except:
        base_score = 0.5
        weighted_score = base_score * weight
        feedback = "Unable to evaluate. Please try again."
    
    # Store evaluation
    state["evaluation_scores"].append({
        "question_num": state["questions_count"],
        "subject": state["current_topic"],
        "difficulty": state["current_difficulty"],
        "base_score": base_score,
        "weighted_score": weighted_score,
        "weight": weight,
        "feedback": feedback,
        "question": state["current_question"],
        "answer": state["user_answer"]
    })
    
    # Track failed topics
    if base_score < 0.4:
        topic_key = f"{state['current_topic']}_{state['current_difficulty']}"
        if topic_key not in state["failed_topics"]:
            state["failed_topics"].append(topic_key)
    
    # Continue or finish
    if state["questions_count"] < 10:
        state["current_stage"] = "questioning"
    else:
        state["current_stage"] = "final_evaluation"
    
    return state

def main():
    print("\n=== VOICE-BASED TECHNICAL INTERVIEW SYSTEM ===\n")
    
    # Load vectorstores
    python_path = r"C:\Users\abina\Desktop\Resume Tracker\Interview_prep\python_book.pdf"
    sql_path = r"C:\Users\abina\Desktop\Resume Tracker\Interview_prep\sql_book.pdf"
    python_vs, sql_vs = load_vectorstores(python_path, sql_path)
    
    # Greeting
    greeting_text = "Hello! Welcome to the technical interview. May I know your name?"
    print(f"[TTS] {greeting_text}")
    with HiddenPrints():
        audio.speak(greeting_text)
    
    user_name_raw = audio.listen(timeout=30, phrase_time_limit=15)
    
    if user_name_raw:
        user_name_lower = user_name_raw.lower()
        
        # Remove greetings
        for greeting in ["hi", "hello", "hey"]:
            user_name_lower = user_name_lower.replace(greeting, "")
        
        user_name_lower = user_name_lower.replace(",", "").strip()
        
        # Extract name
        if "my name is" in user_name_lower:
            user_name = user_name_lower.split("my name is")[1].strip()
        elif "i am" in user_name_lower:
            user_name = user_name_lower.split("i am")[1].strip()
        elif "i'm" in user_name_lower:
            user_name = user_name_lower.split("i'm")[1].strip()
        elif "this is" in user_name_lower:
            user_name = user_name_lower.split("this is")[1].strip()
        else:
            user_name = user_name_lower
        
        user_name = user_name.replace("my name", "").replace("i am", "").replace("i'm", "").strip()
        user_name = user_name.title()
        print(f"[STT] {user_name}\n")
    else:
        user_name = "Guest"
        print(f"[STT] Guest\n")
    
    # Ready check
    begin_text = f"Hello {user_name}! Let's begin our interview. Are you ready?"
    print(f"[TTS] {begin_text}")
    with HiddenPrints():
        audio.speak(begin_text)
    
    ready_response = audio.listen(timeout=30, phrase_time_limit=10)
    if ready_response:
        print(f"[STT] {ready_response}\n")
    
    time.sleep(1)
    
    # Introduction
    intro_text = "Please tell us about yourself, your background, and your experience with Python and SQL."
    print(f"[TTS] {intro_text}")
    with HiddenPrints():
        audio.speak(intro_text)
    
    user_intro = audio.listen_long_form(timeout=15, phrase_time_limit=60)
    
    if user_intro:
        print(f"[STT] {user_intro}\n")
    else:
        user_intro = "No introduction provided"
        print(f"[STT] No introduction provided\n")
    
    # Skill assessment
    python_rating = 5
    sql_rating = 5
    
    if user_intro and user_intro != "No introduction provided":
        try:
            # Use keyword-based parsing (more reliable than LLM)
            intro_lower = user_intro.lower()
            
            # Define skill level keywords
            advanced_keywords = ["expert", "advanced", "advance", "strong", "proficient", "very good", "excellent", "10"]
            intermediate_keywords = ["good", "comfortable", "intermediate", "decent", "okay", "fair", "6", "7"]
            beginner_keywords = ["basic", "beginner", "learning", "new", "weak", "starting", "3", "4"]
            
            # Check for "both" statements first
            both_patterns = ["both", "in python and sql", "python and sql", "10 python and sql", "advance 10"]
            if any(pattern in intro_lower for pattern in both_patterns):
                # User mentioned both together - apply same rating
                if any(word in intro_lower for word in advanced_keywords):
                    python_rating, sql_rating = 9, 9
                elif any(word in intro_lower for word in intermediate_keywords):
                    python_rating, sql_rating = 6, 6
                elif any(word in intro_lower for word in beginner_keywords):
                    python_rating, sql_rating = 3, 3
                else:
                    python_rating, sql_rating = 5, 5
            else:
                # Parse Python skill separately
                python_rating = 5  # default
                if "python" in intro_lower:
                    # Find the skill level mentioned near "python"
                    python_index = intro_lower.find("python")
                    context = intro_lower[max(0, python_index-30):python_index+30]
                    
                    if any(word in context for word in advanced_keywords):
                        python_rating = 9
                    elif any(word in context for word in intermediate_keywords):
                        python_rating = 6
                    elif any(word in context for word in beginner_keywords):
                        python_rating = 3
                
                # Parse SQL skill separately
                sql_rating = 5  # default
                if "sql" in intro_lower:
                    # Find the skill level mentioned near "sql"
                    sql_index = intro_lower.find("sql")
                    context = intro_lower[max(0, sql_index-30):sql_index+30]
                    
                    if any(word in context for word in advanced_keywords):
                        sql_rating = 9
                    elif any(word in context for word in intermediate_keywords):
                        sql_rating = 6
                    elif any(word in context for word in beginner_keywords):
                        sql_rating = 3
            
            print(f"[SYSTEM] Detected - Python: {python_rating}/10, SQL: {sql_rating}/10\n")
            
        except Exception as e:
            python_rating = 5
            sql_rating = 5
            print(f"[SYSTEM] Using default - Python: 5/10, SQL: 5/10\n")
    
    # Initialize state
    state = {
        "current_stage": "questioning",
        "user_name": user_name,
        "user_intro": user_intro,
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
        "interview_complete": False,
        "asked_question_texts": []
    }
    
    print("=== TECHNICAL QUESTIONS ===\n")
    
    # Interview loop
    while state["questions_count"] < 10:
        state = question_agent(state, python_vs, sql_vs)
        
        question = state["current_question"]
        difficulty = state["current_difficulty"]
        subject = state["current_topic"]
        
        # Display question WITHOUT difficulty level (user shouldn't know)
        print(f"\n[Question {state['questions_count']}/10 - {subject}]")
        print(f"[TTS] {question}")
        
        with HiddenPrints():
            audio.speak(question)
        
        answer = audio.listen_long_form(timeout=12, phrase_time_limit=50)
        
        if answer:
            print(f"[STT] {answer}")
        else:
            print(f"[STT] No answer provided")
            answer = "No answer provided (timeout)"
        
        state["user_answer"] = answer
        state["questions_asked"].append({
            "question": question,
            "subject": subject,
            "difficulty": difficulty,
            "answer": state["user_answer"]
        })
        
        # Evaluate answer immediately (but don't display score yet)
        state = evaluation_agent(state, python_vs, sql_vs)
        
        # Internal tracking only - for difficulty adjustment
        current_score = state["evaluation_scores"][-1]["base_score"] * 10
        print(f"[SYSTEM] Evaluated internally - adjusting next difficulty...")
        
        time.sleep(1)
    
    # Final results
    print("\n" + "="*60)
    print("=== FINAL RESULTS ===")
    print("="*60 + "\n")
    
    if state["evaluation_scores"]:
        total_weighted = sum([s["weighted_score"] for s in state["evaluation_scores"]])
        total_possible = sum([s["weight"] for s in state["evaluation_scores"]])
        
        overall_score = (total_weighted / total_possible) * 100 if total_possible > 0 else 0
        
        print(f"Overall Score: {overall_score:.1f}/100\n")
        
        print("Detailed Question Breakdown:")
        print("-" * 60)
        for s in state["evaluation_scores"]:
            score_10_scale = s['base_score'] * 10  # Convert to 0-10 scale
            print(f"Q{s['question_num']}: {s['subject']} ({s['difficulty'].upper()})")
            print(f"   Question: {s['question']}")
            print(f"   Your Answer: {s['answer'][:80]}{'...' if len(s['answer']) > 80 else ''}")
            print(f"   Score: {score_10_scale:.1f}/10 (Base: {s['base_score']*100:.0f}%, Weighted: {s['weighted_score']:.2f})")
            print(f"   Feedback: {s['feedback']}")
            print()
        print("-" * 60 + "\n")
        
        # Calculate subject averages
        python_scores = [s for s in state["evaluation_scores"] if s["subject"] == "Python"]
        sql_scores = [s for s in state["evaluation_scores"] if s["subject"] == "SQL"]
        
        if python_scores:
            python_avg = sum([s["base_score"] for s in python_scores]) / len(python_scores) * 100
            print(f"Python Average: {python_avg:.1f}/100 ({len(python_scores)} questions)")
        
        if sql_scores:
            sql_avg = sum([s["base_score"] for s in sql_scores]) / len(sql_scores) * 100
            print(f"SQL Average: {sql_avg:.1f}/100 ({len(sql_scores)} questions)")
        
        print()
        
        grade = ("A+" if overall_score >= 90 else "A" if overall_score >= 80 else
                 "B" if overall_score >= 70 else "C" if overall_score >= 60 else
                 "D" if overall_score >= 50 else "F")
        
        print(f"Final Grade: {grade}\n")
        print("="*60)
        
        summary = f"You've completed all 10 questions with a grade of {grade}."
        
        thankyou_text = f"Thank you {user_name} for completing the interview. Your overall score is {overall_score:.0f} out of 100. {summary}"
        print(f"\n[TTS] {thankyou_text}\n")
        with HiddenPrints():
            audio.speak(thankyou_text)
        
        print("\n=== Interview Complete! ===\n")

if __name__ == "__main__":
    main()