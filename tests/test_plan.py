"""
tests/test_plan.py
===================
Dedicated tests for the StudyPlanGenerator class.

Tests the class directly (unit tests) without going through HTTP.
This is faster and catches bugs at the Python level before testing
the HTTP layer.

Run:
    pytest tests/test_plan.py -v
"""

import pytest
from app.recommenders.study_plan import StudyPlanGenerator, TOPIC_MAP, KEYWORD_DOMAIN_MAP


@pytest.fixture(scope="module")
def gen():
    """Shared StudyPlanGenerator instance for all tests in this module."""
    return StudyPlanGenerator()


class TestKeywordExtraction:

    def test_extracts_keywords_from_gate_goal(self, gen):
        kws = gen._extract_keywords("prepare for GATE exam")
        assert len(kws) > 0
        # At least one keyword should be meaningful
        assert any(k in ["gate", "prepare", "exam"] for k in kws)

    def test_extracts_keywords_from_dsa_goal(self, gen):
        kws = gen._extract_keywords("improve DSA skills for placements")
        assert len(kws) > 0

    def test_removes_stopwords(self, gen):
        kws = gen._extract_keywords("the and for in a an")
        # All common stopwords should be removed
        assert len(kws) == 0

    def test_returns_max_5_keywords(self, gen):
        kws = gen._extract_keywords("machine learning deep learning neural networks python data science")
        assert len(kws) <= 5

    def test_handles_empty_string(self, gen):
        kws = gen._extract_keywords("")
        assert isinstance(kws, list)

    def test_handles_single_word(self, gen):
        kws = gen._extract_keywords("python")
        assert isinstance(kws, list)


class TestDomainDetection:

    DOMAIN_CASES = [
        ("prepare for GATE exam",           "gate"),
        ("get a web development internship","web"),
        ("crack Amazon SDE interview",      "amazon"),
        ("learn machine learning",          "ml"),
        ("improve DSA for placement",       "dsa"),
        ("build python backend project",    "python"),
        ("study cloud computing and AWS",   "cloud"),
        ("data science analytics project",  "data_science"),
        ("system design for interviews",    "system_design"),
        ("competitive programming Codeforces", "competitive"),
    ]

    @pytest.mark.parametrize("goal,expected_domain", DOMAIN_CASES)
    def test_detects_correct_domain(self, gen, goal, expected_domain):
        kws    = gen._extract_keywords(goal)
        domain = gen._detect_domain(goal, kws)
        assert domain == expected_domain, \
            f"Goal '{goal}': expected domain '{expected_domain}', got '{domain}'"

    def test_unknown_goal_falls_back_to_default(self, gen):
        kws    = gen._extract_keywords("learn to cook Italian pasta")
        domain = gen._detect_domain("learn to cook Italian pasta", kws)
        assert domain == "default"

    def test_all_domains_in_topic_map(self):
        """Every domain value in KEYWORD_DOMAIN_MAP must exist as a key in TOPIC_MAP."""
        all_domains = set(KEYWORD_DOMAIN_MAP.values())
        for domain in all_domains:
            assert domain in TOPIC_MAP, f"Domain '{domain}' in KEYWORD_DOMAIN_MAP but not in TOPIC_MAP"


class TestScheduleGeneration:

    def test_schedule_has_correct_length(self, gen):
        result = gen.generate("prepare for GATE", days_available=14)
        assert len(result["schedule"]) == 14

    def test_schedule_7_days(self, gen):
        result = gen.generate("learn python", days_available=7)
        assert len(result["schedule"]) == 7

    def test_schedule_60_days(self, gen):
        result = gen.generate("crack placement", days_available=60)
        assert len(result["schedule"]) == 60

    def test_every_day_has_required_keys(self, gen):
        result   = gen.generate("prepare for GATE", days_available=14)
        required = {"day", "topic", "subtopics", "hours", "day_type"}
        for day in result["schedule"]:
            assert required.issubset(set(day.keys())), \
                f"Day {day['day']} missing keys: {required - set(day.keys())}"

    def test_day_numbers_are_sequential(self, gen):
        result = gen.generate("learn DSA", days_available=10)
        numbers = [d["day"] for d in result["schedule"]]
        assert numbers == list(range(1, 11))

    def test_last_day_is_always_revision(self, gen):
        for days in [7, 14, 30]:
            result   = gen.generate("prepare for gate", days_available=days)
            last_day = result["schedule"][-1]
            assert last_day["day_type"] == "revision", \
                f"Last day of {days}-day plan should be 'revision', got '{last_day['day_type']}'"

    def test_seventh_days_are_revision(self, gen):
        result = gen.generate("prepare for gate", days_available=28)
        for day in result["schedule"]:
            if day["day"] % 7 == 0 and day["day"] < 28:
                assert day["day_type"] == "revision", \
                    f"Day {day['day']} should be revision"

    def test_hours_per_day_respected(self, gen):
        result = gen.generate("learn python", days_available=7, hours_per_day=5)
        study_days = [d for d in result["schedule"] if d["day_type"] == "study"]
        for day in study_days:
            assert day["hours"] == 5

    def test_subtopics_is_list(self, gen):
        result = gen.generate("learn DSA", days_available=7)
        for day in result["schedule"]:
            assert isinstance(day["subtopics"], list)
            assert len(day["subtopics"]) > 0

    def test_hours_is_positive_integer(self, gen):
        result = gen.generate("prepare for gate", days_available=14)
        for day in result["schedule"]:
            assert isinstance(day["hours"], int)
            assert day["hours"] >= 1


class TestPlanMetadata:

    def test_result_contains_detected_keywords(self, gen):
        result = gen.generate("prepare for GATE exam", days_available=7)
        assert "detected_keywords" in result
        assert isinstance(result["detected_keywords"], list)

    def test_result_contains_detected_domain(self, gen):
        result = gen.generate("prepare for GATE exam", days_available=7)
        assert "detected_domain" in result
        assert result["detected_domain"] in TOPIC_MAP

    def test_total_days_matches_schedule_length(self, gen):
        result = gen.generate("learn python", days_available=21)
        assert result["total_days"] == 21
        assert len(result["schedule"]) == 21

    def test_total_hours_is_sum_of_schedule_hours(self, gen):
        result = gen.generate("crack Amazon", days_available=14)
        expected_hours = sum(d["hours"] for d in result["schedule"])
        assert result["total_hours"] == expected_hours

    def test_total_topics_matches_unique_topics(self, gen):
        result = gen.generate("learn DSA", days_available=14)
        unique_topics = len(set(d["topic"] for d in result["schedule"]))
        assert result["total_topics"] == unique_topics


class TestTopicMap:

    def test_all_domains_have_at_least_4_topics(self):
        for domain, topics in TOPIC_MAP.items():
            assert len(topics) >= 4, \
                f"Domain '{domain}' has only {len(topics)} topics (minimum 4)"

    def test_no_empty_topic_strings(self):
        for domain, topics in TOPIC_MAP.items():
            for topic in topics:
                assert topic.strip(), f"Empty topic in domain '{domain}'"

    def test_default_domain_exists(self):
        assert "default" in TOPIC_MAP

    def test_all_major_domains_present(self):
        required = {"gate", "dsa", "web", "ml", "python", "cloud", "amazon"}
        for domain in required:
            assert domain in TOPIC_MAP, f"Missing required domain: '{domain}'"
