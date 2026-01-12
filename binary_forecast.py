import asyncio
import datetime

import numpy as np

from models import BinaryForecast


BINARY_PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{title}

Question background:
{background}


This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{resolution_criteria}

{fine_print}

Today is {today}.

Use web search to find relevant information about this question. Specifically:
- Search for similar or exact markets on Polymarket (polymarket.com) and Kalshi (kalshi.com)
- Anchor your forecast to the current market prices on these platforms
- Consider recent news and developments that might affect the outcome
- Cite your sources when using web search results

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) A brief description of a scenario that results in a No outcome.
(d) A brief description of a scenario that results in a Yes outcome.

You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

Provide your final probability as a percentage (0-100) and detailed reasoning.
"""


async def get_binary_gpt_prediction(
    question_details: dict,
    num_runs: int,
    call_llm_with_structured_output,
) -> tuple[float, str]:
    """
    Generate a binary forecast prediction using structured outputs.
    
    Args:
        question_details: Dictionary containing question information
        num_runs: Number of runs to aggregate (median is taken)
        call_llm_with_structured_output: Function to call LLM with structured outputs
        
    Returns:
        Tuple of (probability as float 0-1, comment string)
    """
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    question_type = question_details["type"]

    content = BINARY_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
    )

    async def get_rationale_and_probability(content: str) -> tuple[float, str]:
        forecast, reasoning = await call_llm_with_structured_output(
            content, BinaryForecast
        )

        probability = forecast.probability_percent
        probability = min(99, max(1, probability))  # clamp between 1 and 99
        comment = (
            f"Extracted Probability: {probability}%\n\nReasoning: "
            f"{reasoning}\n\n\n"
        )
        return probability, comment

    probability_and_comment_pairs = await asyncio.gather(
        *[get_rationale_and_probability(content) for _ in range(num_runs)]
    )
    comments = [pair[1] for pair in probability_and_comment_pairs]
    final_comment_sections = [
        f"## Rationale {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]
    probabilities = [pair[0] for pair in probability_and_comment_pairs]
    median_probability = float(np.median(probabilities)) / 100

    final_comment = f"Median Probability: {median_probability}\n\n" + "\n\n".join(
        final_comment_sections
    )
    return median_probability, final_comment
