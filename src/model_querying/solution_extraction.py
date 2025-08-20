import json
import logging
import math
from copy import copy

logger = logging.getLogger(__name__)


def get_yes_no(x):
    x = x.lower()
    y = ("true" in x) or (" yes" in x) or (x == "yes")  # optional: include yes/no
    n = ("false" in x) or (" no" in x) or (x == "no")
    #y = "true" in x
    #n = "false" in x
    if y == n:
        return None
    return y


def get_yes_no_diff_logprobs(logprobs):
    eps = 1e-5
    prob_sums = {False: eps, True: eps}
    for k, v in logprobs.items():
        o = get_yes_no(k)
        if o is None:
            continue
        prob_sums[o] += math.exp(v)

    if prob_sums[False] == eps and prob_sums[True] == eps:
        return 0
    else:
        return math.log(prob_sums[True]) - math.log(prob_sums[False])

def _uid_safe(resp: dict) -> str:
    return (resp.get("metadata") or {}).get("uid", "<unknown>")

def _first_step_logprob_map(resp: dict) -> dict[str, float] | None:
    """
    Normalize various provider/adapter shapes to a dict[token] -> logprob
    for the FIRST decoding step.
    Supported shapes:
      A) resp["response"]["logprobs"][0] is already a dict[token]->logprob
      B) resp["response"]["logprobs"][0]["top_logprobs"] is a list[{token,logprob}]
      C) resp["logprobs"][0]["top_logprobs"] (top-level)
      D) resp["response"]["logprobs"] is a list of per-step dicts as in Ollama
    """
    # 1) nested under 'response'
    inner = resp.get("response")
    if isinstance(inner, dict) and "logprobs" in inner:
        lp0 = inner["logprobs"][0] if inner["logprobs"] else None
        if lp0 is None:
            return None
        # A) already a dict
        if isinstance(lp0, dict) and "top_logprobs" not in lp0:
            # assume dict[token]->logprob
            return lp0
        # B) ollama-like step with top_logprobs list
        if isinstance(lp0, dict) and "top_logprobs" in lp0 and isinstance(lp0["top_logprobs"], list):
            return {d["token"]: d["logprob"] for d in lp0["top_logprobs"] if "token" in d and "logprob" in d}

    # 2) top-level 'logprobs'
    lp = resp.get("logprobs")
    if isinstance(lp, list) and lp:
        lp0 = lp[0]
        if isinstance(lp0, dict) and "top_logprobs" in lp0 and isinstance(lp0["top_logprobs"], list):
            return {d["token"]: d["logprob"] for d in lp0["top_logprobs"] if "token" in d and "logprob" in d}
        if isinstance(lp0, dict) and "top_logprobs" not in lp0:
            # Some adapters might already give a dict
            return lp0

    return None

def _extract_score_safe(resp: dict) -> float:
    logprob_map = _first_step_logprob_map(resp)
    if not logprob_map:
        return 0.0
    try:
        return get_yes_no_diff_logprobs(logprob_map)
    except Exception as e:
        logger.info(f"Problem {_uid_safe(resp)}: error computing yes/no diff: {repr(e)}")
        return 0.0

"""
def extract_claim_logprobs(response):
    response = response.copy()
    try:
        logprobs = response["response"]["logprobs"][0]
        response[f"score"] = get_yes_no_diff_logprobs(logprobs)
    except Exception as e:
        logger.info(
            f"Problem {response['metadata']['uid']}: Error extracting judgment: {repr(e)}"
        )
        response["score"] = 0
    return response

def extract_decision_logprobs(response):
    response = response.copy()
    try:
        logprobs = response["response"]["logprobs"][0]
        response[f"score"] = get_yes_no_diff_logprobs(logprobs)
    except Exception as e:
        logger.info(
            f"Problem {response['metadata']['uid']}: Error extracting decision: {repr(e)}"
        )
        response["score"] = 0
    return response
"""

def extract_claim_logprobs(response: dict) -> dict:
    resp = response.copy()
    try:
        resp["score"] = _extract_score_safe(resp)
    except Exception as e:
        logger.info(f"Problem {_uid_safe(resp)}: Error extracting judgment: {repr(e)}")
        resp["score"] = 0.0
    return resp

def extract_decision_logprobs(response: dict) -> dict:
    resp = response.copy()
    try:
        resp["score"] = _extract_score_safe(resp)
    except Exception as e:
        logger.info(f"Problem {_uid_safe(resp)}: Error extracting decision: {repr(e)}")
        resp["score"] = 0.0
    return resp
