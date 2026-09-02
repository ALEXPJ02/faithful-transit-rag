"""The evaluation harness — the object of study, not a side quest.

Retrieval precision/recall, answer faithfulness and hallucination rate over
a held-out QA set, scored with Ragas plus a custom LLM-as-judge, and
compared against a static-retrieval-only baseline. Prediction-grounded
answers need their own treatment: 'faithful' there means the stated error
margin matches the model's measured MAE, not that a claim appears verbatim
in a retrieved passage.
"""
