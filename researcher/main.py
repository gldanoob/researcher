import asyncio

from researcher.agents.literature_review import literature_review

if __name__ == "__main__":
    with open("data/proposal.typ", "r") as f:
        proposal_text = f.read()
    with open("data/works.bib", "r") as f:
        bib_text = f.read()

    asyncio.run(literature_review(
        f"""Conduct PHD level research on the topic in the following proposal:
{proposal_text}
{bib_text}
"""
    ))
