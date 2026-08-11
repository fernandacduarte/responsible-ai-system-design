from __future__ import annotations

import unittest

from antispoiler.book import Chunk
from antispoiler.discuss import discuss_response, format_validation_context


class FakeIndex:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, k):
        return list(range(min(k, len(self.chunks))))


class FakeLLM:
    def __init__(self):
        self.system = ""
        self.user = ""
        self.max_tokens = None

    def complete(self, system, user, max_tokens=1024):
        self.system = system
        self.user = user
        self.max_tokens = max_tokens
        return "A bounded follow-up answer."


def chunk(position: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"ch{position:02d}_p01",
        chapter_index=position,
        chapter_label=f"Chapter {position}",
        paragraph_index=1,
        text=text,
    )


class DiscussResponseTests(unittest.TestCase):
    def test_follow_up_excludes_passages_past_reader_position(self):
        llm = FakeLLM()
        index = FakeIndex(
            [
                chunk(1, "The acknowledged idea appears in the opening."),
                chunk(2, "The narrator treats the statement ironically."),
                chunk(20, "A future event the reader has not reached."),
            ]
        )

        answer = discuss_response(
            llm,
            index,
            selected_text="acknowledged",
            intention="define",
            original_answer="It means recognized as true.",
            validation={"enabled": True, "ui_state": "Valid", "message": "Supported."},
            messages=[{"role": "user", "content": "Why is it important here?"}],
            reader_position=2,
            title="Example Book",
            author="Example Author",
        )

        self.assertEqual(answer, "A bounded follow-up answer.")
        self.assertIn("The acknowledged idea appears", llm.user)
        self.assertIn("The narrator treats", llm.user)
        self.assertNotIn("future event", llm.user)
        self.assertIn("READER: Why is it important here?", llm.user)
        self.assertIn("Never use, mention, hint at", llm.system)
        self.assertEqual(llm.max_tokens, 1200)

    def test_unreliable_claims_are_forwarded_without_evidence_payload(self):
        summary = format_validation_context(
            {
                "enabled": True,
                "ui_state": "Not reliable",
                "message": "One claim was contradicted.",
                "claims": [
                    {
                        "verdict": "Contradicted",
                        "claim": "The character has already left.",
                        "reason": "The passage says the character remains.",
                    }
                ],
                "evidence": [{"text": "A very large payload should not be copied."}],
            }
        )

        self.assertIn("Not reliable", summary)
        self.assertIn("Contradicted", summary)
        self.assertIn("character remains", summary)
        self.assertNotIn("very large payload", summary)

    def test_conversation_must_end_with_reader_question(self):
        with self.assertRaisesRegex(ValueError, "end with a user question"):
            discuss_response(
                FakeLLM(),
                FakeIndex([chunk(1, "Available text")]),
                selected_text="text",
                intention="paraphrase",
                original_answer="Plain text.",
                validation=None,
                messages=[{"role": "assistant", "content": "Previous answer"}],
                reader_position=1,
                title="Example",
                author="",
            )


if __name__ == "__main__":
    unittest.main()
