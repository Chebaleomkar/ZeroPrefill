"""End-of-session save: does the next turn reuse the model's last answer?

  0. unit tests on vLLM 0.30.0 (the version the connector targets)
  1. turn 1, connector on, empty cache -> prompt + a long answer saved
  2. turn 2, fresh server, connector on -> prompt + answer + new question  (warm)
  3. turn 2, fresh plain vLLM server                                       (cold)

Segments hold 1,024 tokens and the 16,031-token prompt ends inside segment 15
(tokens 15,360 to 16,384). A prompt-only save can restore at most 15,360 tokens
of turn 2; any hit above that comes from the saved answer. The answer must be
over ~360 tokens to fill segment 15, so turn 1 asks for a detailed explanation.
(Forcing length with min_tokens made the model emit <tool_call> and the server's
tool parser swallowed the rest of the text.)

Pass: tests pass, turn-2 ZP_HIT above 15,360, warm output == cold output.
"""

import glob
import json
import os
import re
import shutil
import sys

import lib

CACHE = "/root/zp-cache"
ANSWER_TOKENS = 1200
QUESTION = ("Explain in detail, file by file, what this scheduler code does "
            "and how the pieces fit together.")
PROMPT_ONLY_HIT = 15360  # what the old prompt-only save could restore


def hit_tokens(tag):
    hits = lib.grep(f"{lib.OUT}/server_{tag}.log", "ZP_HIT")
    m = re.search(r"ZP_HIT tokens=(\d+)", hits[-1]) if hits else None
    return int(m.group(1)) if m else 0


def main():
    lib.sh("nvidia-smi")
    lib.install_vllm()
    res = {}

    tests = lib.sh(f"cd {lib.CODE_DIR} && {sys.executable} -m pip install -q pytest && "
                   f"{sys.executable} -m pytest tests -q -p no:cacheprovider 2>&1 | tail -5",
                   capture_output=True, text=True)
    res["unit_tests"] = (tests.stdout or "") + (tests.stderr or "")
    lib.log(f"unit tests: {res['unit_tests'].strip()[-300:]}")

    turn1 = lib.build_messages()
    short_q = "In three sentences, what does this scheduler code do?"
    assert turn1[-1]["content"].endswith(short_q)
    turn1[-1]["content"] = turn1[-1]["content"][:-len(short_q)] + QUESTION
    shutil.rmtree(CACHE, ignore_errors=True)

    srv = lib.start_server("turn1", cache_root=CACHE)
    try:
        res["turn1"] = lib.chat(turn1, max_tokens=ANSWER_TOKENS)
    finally:
        lib.stop_server(srv)  # waits for ZP_FLUSHED on every rank
    res["zp_saved"] = lib.grep(f"{lib.OUT}/server_turn1.log", "ZP_SAVED")
    res["zp_flushed"] = lib.grep(f"{lib.OUT}/server_turn1.log", "ZP_FLUSHED")
    res["cache_mib"] = sum(os.path.getsize(p) for p in glob.glob(f"{CACHE}/**/*", recursive=True)
                           if os.path.isfile(p)) / 2**20

    turn2 = turn1 + [{"role": "assistant", "content": res["turn1"]["text"]},
                     {"role": "user", "content": "Now list the three most important "
                      "functions in that code, one line each."}]

    srv = lib.start_server("warm", cache_root=CACHE)
    try:
        res["warm"] = lib.chat(turn2)
    finally:
        lib.stop_server(srv)
    res["zp_hit"] = lib.grep(f"{lib.OUT}/server_warm.log", "ZP_HIT")
    res["warm_hit_tokens"] = hit_tokens("warm")

    srv = lib.start_server("cold", prefix_caching=False)
    try:
        res["cold"] = lib.chat(turn2)
    finally:
        lib.stop_server(srv)

    res["turn1_prompt_tokens"] = (res["turn1"]["usage"] or {}).get("prompt_tokens")
    res["turn1_answer_tokens"] = (res["turn1"]["usage"] or {}).get("completion_tokens")
    res["turn2_prompt_tokens"] = (res["warm"]["usage"] or {}).get("prompt_tokens")
    res["answer_reused"] = res["warm_hit_tokens"] > PROMPT_ONLY_HIT
    res["warm_equals_cold"] = res["warm"]["text"] == res["cold"]["text"]
    res["zp_warnings"] = [line for tag in ("turn1", "warm")
                          for line in lib.grep(f"{lib.OUT}/server_{tag}.log", "[ZP]")
                          if "could not place" in line or "mismatch" in line
                          or "final save failed" in line]
    lib.save_json("answer.json", res)

    lib.log(f"turn 1: prompt {res['turn1_prompt_tokens']} + answer {res['turn1_answer_tokens']} tokens")
    lib.log(f"turn 2: prompt {res['turn2_prompt_tokens']} tokens, ZP_HIT {res['warm_hit_tokens']} "
            f"(prompt-only save would give {PROMPT_ONLY_HIT}); answer reused: {res['answer_reused']}")
    lib.log(f"turn 2 ttft: cold {res['cold']['ttft_s']}s  warm {res['warm']['ttft_s']}s")
    lib.log(f"turn 2 energy: cold {res['cold']['energy_j']}J  warm {res['warm']['energy_j']}J")
    lib.log(f"warm == cold: {res['warm_equals_cold']}  warnings: {res['zp_warnings']}")
    lib.log(f"flushed: {json.dumps(res['zp_flushed'][-2:])[:400]}")


if __name__ == "__main__":
    main()
