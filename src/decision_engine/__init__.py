"""Zendesk ticket triage decision engine (Module 2).

Reads mapped ticket documents from Mongo, asks an LLM (via Cheaper Inference)
for one of five triage classifications, and appends the result to
tickets.agent_reviews. Never writes to Zendesk; write-back is Module 3.
"""
