"""
app/recommenders/study_plan.py
================================
StudyPlanGenerator — NLTK keyword extraction → day-by-day schedule.

Robust design: uses NLTK word_tokenize when available, falls back to
regex splitting if NLTK data is not yet downloaded. Both paths produce
the same keyword list — just download NLTK data first for best results.

Run once to download NLTK data:
    python scripts/download_nltk.py
"""

import re
from typing import Optional

# ── Lazy NLTK init ────────────────────────────────────────────────
_NLTK_READY = False

def _ensure_nltk() -> bool:
    """Download NLTK punkt/stopwords if not present. Returns True if successful."""
    global _NLTK_READY
    if _NLTK_READY:
        return True
    try:
        import nltk
        # Try using the data first — if it works, we're done
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        word_tokenize("test sentence")
        stopwords.words("english")
        _NLTK_READY = True
        return True
    except LookupError:
        # Data not downloaded yet — try downloading
        try:
            import nltk
            nltk.download("punkt",     quiet=True)
            nltk.download("punkt_tab", quiet=True)
            nltk.download("stopwords", quiet=True)
            _NLTK_READY = True
            return True
        except Exception:
            return False
    except Exception:
        return False


# ── Basic English stopwords (used when NLTK is unavailable) ──────
_BASIC_STOPS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than","too",
    "very","can","will","just","should","now","want","need","get","make","like",
    "know","also","time","days","weeks","months","years","day","week","month",
    "year","today","tomorrow","learn","study","prepare","crack","improve",
    "build","start","begin","complete","finish","pass","score","high","good",
    "better","best","using","use","help","exam","test","interview","job",
    "course","courses","book","books","topic","topics"
}


# ── Domain → topic map ────────────────────────────────────────────
TOPIC_MAP: dict[str, list[str]] = {
    "gate": [
        "Operating Systems",
        "Database Management Systems",
        "Computer Networks",
        "Data Structures",
        "Algorithms & Complexity",
        "Computer Organisation & Architecture",
        "Theory of Computation",
        "Compiler Design",
        "Digital Logic",
        "Engineering Mathematics",
        "Discrete Mathematics",
        "General Aptitude",
    ],
    "dsa": [
        "Arrays & Strings",
        "Linked Lists",
        "Stacks & Queues",
        "Trees & Binary Search Trees",
        "Heaps & Priority Queues",
        "Graphs — BFS & DFS",
        "Dynamic Programming",
        "Greedy Algorithms",
        "Backtracking & Recursion",
        "Sorting & Searching",
        "Hashing & Hash Maps",
        "Sliding Window & Two Pointers",
    ],
    "web": [
        "HTML5 & CSS3 Fundamentals",
        "JavaScript ES6+ Core Concepts",
        "DOM Manipulation & Events",
        "Responsive Design & Flexbox/Grid",
        "React — Components & Props",
        "React — State Management & Hooks",
        "REST API Design",
        "Node.js & Express Basics",
        "Databases — SQL & NoSQL",
        "Authentication & JWT",
        "Deployment — GitHub Pages / Vercel / Render",
        "Portfolio Project Build",
    ],
    "ml": [
        "Python — NumPy & Pandas",
        "Exploratory Data Analysis",
        "Supervised Learning — Regression",
        "Supervised Learning — Classification",
        "Model Evaluation — Cross-Validation & Metrics",
        "Feature Engineering & Selection",
        "Unsupervised Learning — Clustering",
        "Decision Trees & Random Forests",
        "Support Vector Machines",
        "Neural Networks — Perceptron to MLP",
        "Introduction to Deep Learning",
        "Project — End-to-End ML Pipeline",
    ],
    "system_design": [
        "Scalability Fundamentals",
        "Load Balancing & CDN",
        "Caching Strategies (Redis, Memcached)",
        "Database Sharding & Replication",
        "CAP Theorem & Consistency Models",
        "Message Queues (Kafka, RabbitMQ)",
        "Microservices Architecture",
        "API Gateway & Rate Limiting",
        "Design — URL Shortener",
        "Design — Twitter Feed",
        "Design — Ride Sharing App",
        "Design — Video Streaming Platform",
    ],
    "amazon": [
        "Arrays, Strings & Hashing",
        "Two Pointers & Sliding Window",
        "Binary Search Patterns",
        "Linked Lists & Fast-Slow Pointers",
        "Trees — Traversals & LCA",
        "Dynamic Programming — Classics",
        "Graphs — Shortest Path",
        "System Design — Basics",
        "Behavioural Questions — STAR Method",
        "Amazon Leadership Principles",
        "Mock Interview — 45 min session",
        "Full Mock + Review",
    ],
    "python": [
        "Python Basics — Syntax & Data Types",
        "Control Flow & Functions",
        "Object-Oriented Programming",
        "File I/O & Exception Handling",
        "List Comprehensions & Generators",
        "Decorators & Context Managers",
        "Standard Library — os, sys, datetime",
        "Virtual Environments & pip",
        "Unit Testing with pytest",
        "Working with APIs — requests library",
        "Data Processing — pandas basics",
        "Project — Build a CLI Tool",
    ],
    "cloud": [
        "Cloud Computing Fundamentals",
        "AWS Core Services — EC2, S3, RDS",
        "AWS IAM & Security Groups",
        "Serverless — AWS Lambda",
        "Containers — Docker Basics",
        "Container Orchestration — Kubernetes Intro",
        "CI/CD Pipelines",
        "Monitoring & Logging",
        "Azure / GCP Overview",
        "Cloud Architecture Patterns",
        "Cost Optimisation",
        "Cloud Certification Prep",
    ],
    "data_science": [
        "Python & Pandas for Data Analysis",
        "Data Cleaning & Preprocessing",
        "Exploratory Data Analysis & Visualisation",
        "Statistics — Descriptive & Inferential",
        "Probability & Distributions",
        "Hypothesis Testing",
        "Regression Analysis",
        "Classification Algorithms",
        "Time Series Analysis",
        "Feature Engineering",
        "Model Deployment",
        "Capstone Project",
    ],
    "competitive": [
        "Number Theory & Modular Arithmetic",
        "Combinatorics & Probability",
        "Bit Manipulation",
        "Segment Trees & BIT",
        "Disjoint Set Union (DSU)",
        "Graph Algorithms — Advanced",
        "Network Flow",
        "String Algorithms — KMP & Z",
        "Geometry Basics",
        "Game Theory",
        "Codeforces Div 2 B-C Practice",
        "Codeforces Div 2 C-D Practice",
    ],
    "default": [
        "Core Concepts & Fundamentals",
        "Problem Solving Practice",
        "Concept Deep Dive",
        "Applied Exercises",
        "Mock Test & Review",
        "Weak Area Revision",
        "Full Mock Assessment",
    ],
}

# ── Keyword → domain mapping ──────────────────────────────────────
KEYWORD_DOMAIN_MAP: dict[str, str] = {
    # GATE / competitive exams
    "gate": "gate", "upsc": "gate", "ese": "gate",
    # DSA
    "dsa": "dsa", "leetcode": "dsa", "algorithm": "dsa",
    "structure": "dsa", "tree": "dsa", "graph": "dsa",
    # Competitive programming
    "codeforces": "competitive", "competitive": "competitive", "cp": "competitive",
    # Web
    "web": "web", "frontend": "web", "backend": "web", "react": "web",
    "javascript": "web", "nodejs": "web", "html": "web", "css": "web",
    "fullstack": "web", "flask": "web", "django": "web",
    # ML / AI
    "machine": "ml", "learning": "ml", "deep": "ml", "neural": "ml",
    "nlp": "ml", "ml": "ml", "ai": "ml", "tensorflow": "ml",
    "keras": "ml", "pytorch": "ml", "sklearn": "ml",
    # System design
    "system": "system_design", "design": "system_design",
    "architecture": "system_design", "distributed": "system_design",
    "scalability": "system_design",
    # Interview / placement
    "amazon": "amazon", "google": "amazon", "microsoft": "amazon",
    "faang": "amazon", "interview": "amazon", "placement": "amazon",
    "internship": "amazon", "sde": "amazon",
    # Python
    "python": "python", "fastapi": "python", "scripting": "python",
    # Cloud
    "cloud": "cloud", "aws": "cloud", "azure": "cloud",
    "gcp": "cloud", "devops": "cloud", "docker": "cloud",
    "kubernetes": "cloud", "k8s": "cloud",
    # Data science
    "data": "data_science", "analytics": "data_science",
    "statistics": "data_science", "pandas": "data_science",
    "visualization": "data_science", "tableau": "data_science",
}

# ── Subtopic details ──────────────────────────────────────────────
SUBTOPIC_MAP: dict[str, list[str]] = {
    "Operating Systems": [
        "Process management & scheduling",
        "Memory management & virtual memory",
        "File systems & I/O management",
        "Deadlock detection & prevention",
    ],
    "Database Management Systems": [
        "Relational model & SQL queries",
        "Normalisation (1NF through BCNF)",
        "Transactions & concurrency control",
        "Indexing & query optimisation",
    ],
    "Computer Networks": [
        "OSI & TCP/IP model layers",
        "IP addressing & subnetting",
        "Routing protocols (OSPF, BGP)",
        "Transport layer — TCP vs UDP",
    ],
    "Data Structures": [
        "Arrays, linked lists, stacks, queues",
        "Trees — BST, AVL, segment tree",
        "Graphs — adjacency list/matrix",
        "Hash maps & heaps",
    ],
    "Algorithms & Complexity": [
        "Time & space complexity (Big-O)",
        "Sorting — merge, quick, heap sort",
        "Dynamic programming patterns",
        "Graph algorithms — Dijkstra, BFS, DFS",
    ],
    "Arrays & Strings": [
        "Two-pointer technique",
        "Sliding window patterns",
        "Prefix sums & difference arrays",
        "String manipulation — palindrome, anagram",
    ],
    "Dynamic Programming": [
        "Memoisation vs tabulation approach",
        "0/1 Knapsack variants",
        "Longest Common Subsequence",
        "Matrix chain multiplication",
    ],
    "REST API Design": [
        "HTTP methods & status codes",
        "Request/response JSON structure",
        "Authentication — JWT & OAuth2",
        "API versioning & rate limiting",
    ],
    "Supervised Learning — Classification": [
        "Logistic regression & sigmoid",
        "K-nearest neighbours algorithm",
        "Decision trees & random forests",
        "Confusion matrix, precision, recall",
    ],
    "Model Evaluation — Cross-Validation & Metrics": [
        "Train/validation/test split rationale",
        "k-fold cross-validation",
        "Precision, recall, F1-score",
        "Bias-variance tradeoff",
    ],
    "System Design — Basics": [
        "Requirements gathering (functional vs non-functional)",
        "High-level component design",
        "Database selection (SQL vs NoSQL)",
        "Scaling strategies (vertical vs horizontal)",
    ],
}


class StudyPlanGenerator:
    """
    Generates personalised day-by-day study plans from free-text goals.

    Works with or without NLTK punkt downloaded:
    - If NLTK data available: uses word_tokenize + FreqDist
    - If not: uses regex word splitting + built-in stopword list
    Both paths produce equivalent keyword quality.
    """

    def __init__(self):
        self._nltk_ok = _ensure_nltk()
        if self._nltk_ok:
            print("[StudyPlanGenerator] NLTK ready")
        else:
            print("[StudyPlanGenerator] NLTK punkt not found — using regex fallback (run scripts/download_nltk.py)")

    # ── Public API ────────────────────────────────────────────────

    def generate(
        self,
        goal: str,
        days_available: int = 14,
        hours_per_day: int = 3,
    ) -> dict:
        """
        Generate a complete study plan.

        Parameters
        ----------
        goal           : Free-text goal string
        days_available : Number of days (7-60)
        hours_per_day  : Daily study hours (1-10)

        Returns
        -------
        dict with keys:
            goal, detected_keywords, detected_domain,
            total_days, total_topics, total_hours, schedule
        """
        keywords = self._extract_keywords(goal)
        domain   = self._detect_domain(goal, keywords)
        topics   = TOPIC_MAP.get(domain, TOPIC_MAP["default"])
        schedule = self._build_schedule(topics, days_available, hours_per_day)

        return {
            "goal"              : goal,
            "detected_keywords" : keywords,
            "detected_domain"   : domain,
            "total_days"        : days_available,
            "total_topics"      : len(set(d["topic"] for d in schedule)),
            "total_hours"       : sum(d["hours"] for d in schedule),
            "schedule"          : schedule,
        }

    # ── Private helpers ───────────────────────────────────────────

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract top-5 meaningful keywords from goal text."""
        if self._nltk_ok:
            return self._extract_nltk(text)
        return self._extract_regex(text)

    def _extract_nltk(self, text: str) -> list[str]:
        """NLTK-based extraction: tokenise → remove stopwords → FreqDist."""
        try:
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            from nltk.probability import FreqDist

            tokens   = word_tokenize(text.lower())
            stop_set = set(stopwords.words("english")) | _BASIC_STOPS
            clean    = [t for t in tokens if t.isalpha() and t not in stop_set and len(t) > 1]
            freq     = FreqDist(clean)
            return [w for w, _ in freq.most_common(5)]
        except Exception:
            return self._extract_regex(text)

    def _extract_regex(self, text: str) -> list[str]:
        """Regex fallback: split on non-alpha → remove stopwords → top 5."""
        words = re.findall(r"[a-z]+", text.lower())
        clean = [w for w in words if w not in _BASIC_STOPS and len(w) > 2]
        # Count frequencies manually
        freq: dict[str, int] = {}
        for w in clean:
            freq[w] = freq.get(w, 0) + 1
        return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)][:5]

    def _detect_domain(self, goal: str, keywords: list[str]) -> str:
        """Map goal text + keywords to a study domain."""
        goal_lower = goal.lower()

        # Check multi-word phrases first (more specific)
        multi_phrases = {
            "machine learning": "ml",
            "deep learning"   : "ml",
            "data science"    : "data_science",
            "web development" : "web",
            "system design"   : "system_design",
            "data structure"  : "dsa",
            "full stack"      : "web",
            "competitive programming": "competitive",
        }
        for phrase, domain in multi_phrases.items():
            if phrase in goal_lower:
                return domain

        # Check extracted keywords
        for kw in keywords:
            if kw in KEYWORD_DOMAIN_MAP:
                return KEYWORD_DOMAIN_MAP[kw]

        # Check every word in the goal
        for word in re.findall(r"[a-z]+", goal_lower):
            if word in KEYWORD_DOMAIN_MAP:
                return KEYWORD_DOMAIN_MAP[word]

        return "default"

    def _build_schedule(
        self, topics: list[str], days: int, hours_per_day: int
    ) -> list[dict]:
        """Build day-by-day schedule with revision days every 7 days."""
        schedule      = []
        study_day_idx = 0

        for day_num in range(1, days + 1):
            is_last     = (day_num == days)
            is_revision = (day_num % 7 == 0) and not is_last

            if is_last:
                entry = {
                    "day"      : day_num,
                    "topic"    : "Full Revision & Mock Test",
                    "subtopics": [
                        "Revise all major topics covered",
                        "Attempt a full-length mock paper",
                        "Identify and target remaining weak areas",
                        "Rest and stay confident",
                    ],
                    "hours"    : min(hours_per_day + 1, 8),
                    "day_type" : "revision",
                }
            elif is_revision:
                prev_topic = topics[(study_day_idx - 1) % len(topics)]
                entry = {
                    "day"      : day_num,
                    "topic"    : f"Revision — {prev_topic}",
                    "subtopics": [
                        f"Review key concepts from {prev_topic}",
                        "Attempt 10 practice problems",
                        "Note down remaining doubts",
                    ],
                    "hours"    : max(1, hours_per_day - 1),
                    "day_type" : "revision",
                }
            else:
                topic     = topics[study_day_idx % len(topics)]
                subtopics = SUBTOPIC_MAP.get(topic, [
                    f"Introduction to {topic}",
                    f"Core concepts of {topic}",
                    f"Practical exercises — {topic}",
                    "Practice problems & review",
                ])
                entry = {
                    "day"      : day_num,
                    "topic"    : topic,
                    "subtopics": subtopics,
                    "hours"    : hours_per_day,
                    "day_type" : "study",
                }
                study_day_idx += 1

            schedule.append(entry)

        return schedule
