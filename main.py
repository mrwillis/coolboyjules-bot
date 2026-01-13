from __future__ import annotations

import asyncio
import json
import os
import time

import dotenv

dotenv.load_dotenv()

import requests
from openai import OpenAI
from pydantic import BaseModel

from binary_forecast import get_binary_gpt_prediction
from multiple_choice_forecast import get_multiple_choice_gpt_prediction
from numeric_forecast import get_numeric_gpt_prediction

"""
This file provides a simple forecasting bot built from the ground up.
We provide this for people who want to dissect
it to build their own bot without using forecasting-tools.

This template uses OpenAI's Responses API with web search and structured outputs.
You will need:
- An OpenAI API key (for the Responses API with web search)
- A Metaculus API key (for posting questions to Metaculus)

The bot uses structured outputs (Pydantic models) to ensure reliable parsing of forecasts,
and leverages OpenAI's web search to find relevant information, including similar markets
on Polymarket and Kalshi to anchor forecasts.

This is not a representative of the template bots used by Metaculus, as there are some
differences in implementation. The actual template bot (e.g. like main.py) has the following differences:
- An LLM now parses the final forecast output (rather than programmatic parsing)
- Support for nominal bounds was added (i.e. when there are discrete questions and normal upper/lower bounds are not as intuitive)
- Upper/Lower bounds are mentioned as suggestions (not ignored) when the bounds are open
- Group questions, conditional questions, and date questions are supported (these types are optional and won't be launched in Spring AIB)
- The research prompt mentions resolution criteria and fine print explicitly

We realize the below code could probably be cleaned up a bit in a few places
Though we are assuming most people will dissect it enough to make this not matter much

Note that this is code is given as-is and though we have have done basic testing
with this file it may be worth double checking key components locally.
"""


######################### CONSTANTS #########################
# Constants
SUBMIT_PREDICTION = True  # set to True to publish your predictions to Metaculus
USE_EXAMPLE_QUESTIONS = False  # set to True to forecast example questions rather than the tournament questions
NUM_RUNS_PER_QUESTION = (
    5  # The median forecast is taken between NUM_RUNS_PER_QUESTION runs
)
SKIP_PREVIOUSLY_FORECASTED_QUESTIONS = True

# Environment variables
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# The tournament IDs below can be used for testing your bot.
Q4_2024_AI_BENCHMARKING_ID = 32506
Q1_2025_AI_BENCHMARKING_ID = 32627
FALL_2025_AI_BENCHMARKING_ID = "fall-aib-2025"
SPRING_2026_AI_BENCHMARKING_ID = "spring-aib-2026"
CURRENT_MINIBENCH_ID = "minibench"

Q4_2024_QUARTERLY_CUP_ID = 3672
Q1_2025_QUARTERLY_CUP_ID = 32630
CURRENT_METACULUS_CUP_ID = None # TBD (Use the slug from the Metaculus Cup URL)

AXC_2025_TOURNAMENT_ID = 32564
AI_2027_TOURNAMENT_ID = "ai-2027"

TOURNAMENT_ID = SPRING_2026_AI_BENCHMARKING_ID

# The example questions can be used for testing your bot. (note that question and post id are not always the same)
EXAMPLE_QUESTIONS = [  # (question_id, post_id)
    (
        578,
        578,
    ),  # Human Extinction - Binary - https://www.metaculus.com/questions/578/human-extinction-by-2100/
    (
        14333,
        14333,
    ),  # Age of Oldest Human - Numeric - https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/
    (
        22427,
        22427,
    ),  # Number of New Leading AI Labs - Multiple Choice - https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/
    (
        38195,
        38880,
    ),  # Number of US Labor Strikes Due to AI in 2029 - Discrete - https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/
]


######################### HELPER FUNCTIONS #########################

# @title Helper functions
AUTH_HEADERS = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}
API_BASE_URL = "https://www.metaculus.com/api"


def post_question_comment(post_id: int, comment_text: str) -> None:
    """
    Post a comment on the question page as the bot user.
    """

    response = requests.post(
        f"{API_BASE_URL}/comments/create/",
        json={
            "text": comment_text,
            "parent": None,
            "included_forecast": True,
            "is_private": True,
            "on_post": post_id,
        },
        **AUTH_HEADERS,  # type: ignore
    )
    if not response.ok:
        raise RuntimeError(response.text)


def post_question_prediction(question_id: int, forecast_payload: dict) -> None:
    """
    Post a forecast on a question.
    """
    url = f"{API_BASE_URL}/questions/forecast/"
    response = requests.post(
        url,
        json=[
            {
                "question": question_id,
                **forecast_payload,
            },
        ],
        **AUTH_HEADERS,  # type: ignore
    )
    print(f"Prediction Post status code: {response.status_code}")
    if not response.ok:
        raise RuntimeError(response.text)


def create_forecast_payload(
    forecast: float | dict[str, float] | list[float],
    question_type: str,
) -> dict:
    """
    Accepts a forecast and generates the api payload in the correct format.

    If the question is binary, forecast must be a float.
    If the question is multiple choice, forecast must be a dictionary that
      maps question.options labels to floats.
    If the question is numeric, forecast must be a dictionary that maps
      quartiles or percentiles to datetimes, or a 201 value cdf.
    """
    if question_type == "binary":
        return {
            "probability_yes": forecast,
            "probability_yes_per_category": None,
            "continuous_cdf": None,
        }
    if question_type == "multiple_choice":
        return {
            "probability_yes": None,
            "probability_yes_per_category": forecast,
            "continuous_cdf": None,
        }
    # numeric or date
    return {
        "probability_yes": None,
        "probability_yes_per_category": None,
        "continuous_cdf": forecast,
    }


def list_posts_from_tournament(
    tournament_id: int | str = TOURNAMENT_ID, offset: int = 0, count: int = 50
) -> list[dict]:
    """
    List (all details) {count} posts from the {tournament_id}
    """
    url_qparams = {
        "limit": count,
        "offset": offset,
        "order_by": "-hotness",
        "forecast_type": ",".join(
            [
                "binary",
                "multiple_choice",
                "numeric",
                "discrete",
            ]
        ),
        "tournaments": [tournament_id],
        "statuses": "open",
        "include_description": "true",
    }
    url = f"{API_BASE_URL}/posts/"
    response = requests.get(url, **AUTH_HEADERS, params=url_qparams)  # type: ignore
    if not response.ok:
        raise Exception(response.text)
    data = json.loads(response.content)
    return data


def get_open_question_ids_from_tournament() -> list[tuple[int, int]]:
    posts = list_posts_from_tournament()

    post_dict = dict()
    for post in posts["results"]:
        if question := post.get("question"):
            # single question post
            post_dict[post["id"]] = [question]

    open_question_id_post_id = []  # [(question_id, post_id)]
    for post_id, questions in post_dict.items():
        for question in questions:
            if question.get("status") == "open":
                print(
                    f"ID: {question['id']}\nQ: {question['title']}\nCloses: "
                    f"{question['scheduled_close_time']}"
                )
                open_question_id_post_id.append((question["id"], post_id))

    print(f"Found {len(open_question_id_post_id)} open questions")

    return open_question_id_post_id


def get_post_details(post_id: int) -> dict:
    """
    Get all details about a post from the Metaculus API.
    """
    url = f"{API_BASE_URL}/posts/{post_id}/"
    print(f"Getting details for {url}")
    response = requests.get(
        url,
        **AUTH_HEADERS,  # type: ignore
    )
    if not response.ok:
        raise Exception(response.text)
    details = json.loads(response.content)
    return details


CONCURRENT_REQUESTS_LIMIT = 5
llm_rate_limiter = asyncio.Semaphore(CONCURRENT_REQUESTS_LIMIT)


######################### LLM CALL FUNCTION #########################

async def call_llm_with_structured_output(
    prompt: str,
    text_format: type[BaseModel],
    model: str = "gpt-5.2-2025-12-11",
    temperature: float = 0.3,
) -> tuple[BaseModel, str]:
    """
    Makes a request to OpenAI's Responses API with web search and structured outputs.
    Returns the parsed Pydantic model and the reasoning text.
    """
    client = OpenAI()
    
    # Print request details
    prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
    print(f"[OpenAI API] Starting call - Model: {model}, Temperature: {temperature}, Format: {text_format.__name__}")
    print(f"[OpenAI API] Prompt preview (first 500 chars): {prompt_preview}")
    print(f"[OpenAI API] Full prompt length: {len(prompt)} characters")
    
    start_time = time.time()

    async with llm_rate_limiter:
        try:
            print("[OpenAI API] Acquired rate limiter semaphore, making API call...")
            
            # Run the synchronous API call in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.responses.parse(
                    model=model,
                    input=[{"role": "user", "content": prompt}],
                    tools=[{"type": "web_search"}],
                    text_format=text_format,
                    temperature=temperature,
                )
            )
            
            elapsed_time = time.time() - start_time
            print(f"[OpenAI API] Call completed in {elapsed_time:.2f} seconds - Status: {response.status}")
            
            # Print response details
            if hasattr(response, 'usage') and response.usage:
                print(f"[OpenAI API] Token usage - Input: {getattr(response.usage, 'input_tokens', 'N/A')}, "
                      f"Output: {getattr(response.usage, 'output_tokens', 'N/A')}, "
                      f"Total: {getattr(response.usage, 'total_tokens', 'N/A')}")
            
            # Print tool calls if any
            if hasattr(response, 'output') and response.output:
                for i, output_item in enumerate(response.output):
                    if hasattr(output_item, 'tool_calls') and output_item.tool_calls:
                        print(f"[OpenAI API] Tool calls in output {i}: {len(output_item.tool_calls)} tool call(s)")
                        for j, tool_call in enumerate(output_item.tool_calls):
                            tool_type = getattr(tool_call, 'type', 'unknown')
                            print(f"[OpenAI API]   Tool call {j}: type={tool_type}")
            
            if response.status == "incomplete":
                incomplete_reason = response.incomplete_details.reason if response.incomplete_details else "unknown"
                print(f"[OpenAI API] WARNING: Response incomplete - Reason: {incomplete_reason}")
                if response.incomplete_details and response.incomplete_details.reason == "max_output_tokens":
                    raise ValueError("Response incomplete: max_output_tokens reached")
                elif response.incomplete_details and response.incomplete_details.reason == "content_filter":
                    raise ValueError("Response incomplete: content_filter triggered")
                else:
                    raise ValueError(f"Response incomplete: {response.incomplete_details}")
            
            # Check for refusals
            if response.output and len(response.output) > 0:
                first_output = response.output[0]
                if hasattr(first_output, 'content') and len(first_output.content) > 0:
                    content = first_output.content[0]
                    if hasattr(content, 'type') and content.type == "refusal":
                        refusal_reason = getattr(content, 'refusal', 'Unknown reason')
                        print(f"[OpenAI API] ERROR: Model refused request - Reason: {refusal_reason}")
                        raise ValueError(f"Model refused: {refusal_reason}")
            
            parsed_output = response.output_parsed
            reasoning = parsed_output.reasoning if hasattr(parsed_output, 'reasoning') else ""
            
            # Print parsed output summary
            if hasattr(parsed_output, '__dict__'):
                output_summary = {k: str(v)[:100] + "..." if len(str(v)) > 100 else str(v) 
                                 for k, v in parsed_output.__dict__.items() if k != 'reasoning'}
                print(f"[OpenAI API] Parsed output summary: {output_summary}")
            
            reasoning_preview = reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
            print(f"[OpenAI API] Reasoning preview (first 200 chars): {reasoning_preview}")
            print(f"[OpenAI API] Full reasoning length: {len(reasoning)} characters")
            
            print(f"[OpenAI API] Successfully parsed {text_format.__name__} output")
            
            return parsed_output, reasoning
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"[OpenAI API] ERROR: Call failed after {elapsed_time:.2f} seconds - Error: {str(e)}")
            raise ValueError(f"Error calling LLM with structured output: {str(e)}")






################### FORECASTING ###################
def forecast_is_already_made(post_details: dict) -> bool:
    """
    Check if a forecast has already been made by looking at my_forecasts in the question data.

    question.my_forecasts.latest.forecast_values has the following values for each question type:
    Binary: [probability for no, probability for yes]
    Numeric: [cdf value 1, cdf value 2, ..., cdf value 201]
    Multiple Choice: [probability for option 1, probability for option 2, ...]
    """
    try:
        forecast_values = post_details["question"]["my_forecasts"]["latest"][
            "forecast_values"
        ]
        return forecast_values is not None
    except Exception:
        return False


async def forecast_individual_question(
    question_id: int,
    post_id: int,
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
) -> str:
    post_details = get_post_details(post_id)
    question_details = post_details["question"]
    title = question_details["title"]
    question_type = question_details["type"]

    summary_of_forecast = ""
    summary_of_forecast += (
        f"-----------------------------------------------\nQuestion: {title}\n"
    )
    summary_of_forecast += f"URL: https://www.metaculus.com/questions/{post_id}/\n"

    if question_type == "multiple_choice":
        options = question_details["options"]
        summary_of_forecast += f"options: {options}\n"
    if (
        forecast_is_already_made(post_details)
        and skip_previously_forecasted_questions == True
    ):
        summary_of_forecast += f"Skipped: Forecast already made\n"
        return summary_of_forecast

    if question_type == "binary":
        forecast, comment = await get_binary_gpt_prediction(
            question_details, num_runs_per_question, call_llm_with_structured_output
        )
    elif question_type == "numeric":
        forecast, comment = await get_numeric_gpt_prediction(
            question_details, num_runs_per_question, call_llm_with_structured_output
        )
    elif question_type == "discrete":
        forecast, comment = await get_numeric_gpt_prediction(
            question_details, num_runs_per_question, call_llm_with_structured_output
        )
    elif question_type == "multiple_choice":
        forecast, comment = await get_multiple_choice_gpt_prediction(
            question_details, num_runs_per_question, call_llm_with_structured_output
        )
    else:
        raise ValueError(f"Unknown question type: {question_type}")

    print(
        f"-----------------------------------------------\nPost {post_id} Question {question_id}:\n"
    )
    print(f"Forecast for post {post_id} (question {question_id}):\n{forecast}")
    print(f"Comment for post {post_id} (question {question_id}):\n{comment}")

    if question_type == "numeric" or question_type == "discrete":
        summary_of_forecast += f"Forecast: {str(forecast)}\n"
    else:
        summary_of_forecast += f"Forecast: {forecast}\n"

    summary_of_forecast += f"Comment:\n```\n{comment}\n```\n\n"

    if submit_prediction == True:
        forecast_payload = create_forecast_payload(forecast, question_type)
        post_question_prediction(question_id, forecast_payload)
        post_question_comment(post_id, comment)
        summary_of_forecast += "Posted: Forecast was posted to Metaculus.\n"

    return summary_of_forecast


async def forecast_questions(
    open_question_id_post_id: list[tuple[int, int]],
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
) -> None:
    print(f"Forecasting {len(open_question_id_post_id)} questions")
    forecast_tasks = [
        forecast_individual_question(
            question_id,
            post_id,
            submit_prediction,
            num_runs_per_question,
            skip_previously_forecasted_questions,
        )
        for question_id, post_id in open_question_id_post_id
    ]
    forecast_summaries = await asyncio.gather(*forecast_tasks, return_exceptions=True)
    print("\n", "#" * 100, "\nForecast Summaries\n", "#" * 100)

    errors = []
    for question_id_post_id, forecast_summary in zip(
        open_question_id_post_id, forecast_summaries
    ):
        question_id, post_id = question_id_post_id
        if isinstance(forecast_summary, Exception):
            print(
                f"-----------------------------------------------\nPost {post_id} Question {question_id}:\nError: {forecast_summary.__class__.__name__} {forecast_summary}\nURL: https://www.metaculus.com/questions/{post_id}/\n"
            )
            errors.append(forecast_summary)
        else:
            print(forecast_summary)

    if errors:
        print("-----------------------------------------------\nErrors:\n")
        error_message = f"Errors were encountered: {errors}"
        print(error_message)
        raise RuntimeError(error_message)


######################## FINAL RUN #########################
if __name__ == "__main__":
    if USE_EXAMPLE_QUESTIONS:
        open_question_id_post_id = EXAMPLE_QUESTIONS
    else:
        open_question_id_post_id = get_open_question_ids_from_tournament()

    asyncio.run(
        forecast_questions(
            open_question_id_post_id,
            SUBMIT_PREDICTION,
            NUM_RUNS_PER_QUESTION,
            SKIP_PREVIOUSLY_FORECASTED_QUESTIONS,
        )
    )
