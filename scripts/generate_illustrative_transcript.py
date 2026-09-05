"""Generate an ILLUSTRATIVE text-to-SQL transcript without calling the LLM.

Why this exists: the live demo (`python -m src.text2sql.demo`) needs
ANTHROPIC_API_KEY. This script produces a transcript that exercises the *real*
validator + read-only executor against the warehouse, so the SQL and result rows
are genuine; only the natural-language phrasing is templated (a stand-in for what
Claude would write). It also demonstrates the retry loop with a deliberately
broken first attempt. Clearly labelled as illustrative in the output.

Run:  python scripts/generate_illustrative_transcript.py
"""
from __future__ import annotations

import pandas as pd

from src.common import config
from src.text2sql.execute_readonly import SQLExecutionError, run_query
from src.text2sql.validate_sql import SQLValidationError

# (question, [sql_attempts...], answer_template)  — last attempt is expected to work.
CASES = [
    (
        "Which specialty prescribed the most Metformin Hcl in CA in the latest year?",
        ["SELECT Prscrbr_Type, SUM(Tot_Clms) AS total FROM prescribers "
         "WHERE Gnrc_Name='Metformin Hcl' AND Prscrbr_State_Abrvtn='CA' "
         "AND Year=(SELECT MAX(Year) FROM prescribers) "
         "GROUP BY Prscrbr_Type ORDER BY total DESC LIMIT 5"],
        lambda df: f"In CA's latest year, **{df.iloc[0]['Prscrbr_Type']}** led Metformin Hcl "
                   f"prescribing with {int(df.iloc[0]['total']):,} claims.",
    ),
    (
        "How did total Amoxicillin claims in TX change between the two most recent years?",
        [   # attempt 1: wrong column name -> triggers the retry loop
            "SELECT Year, SUM(Tot_Clms) AS total FROM prescribers "
            "WHERE Gnrc_Name='Amoxicillin' AND State='TX' GROUP BY Year",
            # attempt 2: corrected
            "SELECT Year, SUM(Tot_Clms) AS total FROM prescribers "
            "WHERE Gnrc_Name='Amoxicillin' AND Prscrbr_State_Abrvtn='TX' "
            "AND Year >= (SELECT MAX(Year)-1 FROM prescribers) GROUP BY Year ORDER BY Year"],
        lambda df: (
            f"Amoxicillin claims in TX went from {int(df.iloc[0]['total']):,} in "
            f"{int(df.iloc[0]['Year'])} to {int(df.iloc[-1]['total']):,} in "
            f"{int(df.iloc[-1]['Year'])} "
            f"({(df.iloc[-1]['total']-df.iloc[0]['total'])/df.iloc[0]['total']*100:+.1f}%)."
        ),
    ),
    (
        "What are the top 5 prescribers by total claims in NY in the latest year?",
        ["SELECT Prscrbr_NPI, SUM(Tot_Clms) AS total FROM prescribers "
         "WHERE Prscrbr_State_Abrvtn='NY' AND Year=(SELECT MAX(Year) FROM prescribers) "
         "GROUP BY Prscrbr_NPI ORDER BY total DESC LIMIT 5"],
        lambda df: f"The top prescriber in NY had {int(df.iloc[0]['total']):,} claims; "
                   f"the top 5 are listed below.",
    ),
    (
        "Compare total claims for each drug class across all states in the latest year.",
        ["SELECT Drug_Class, SUM(Tot_Clms) AS total FROM prescribers "
         "WHERE Year=(SELECT MAX(Year) FROM prescribers) "
         "GROUP BY Drug_Class ORDER BY total DESC"],
        lambda df: f"In the latest year, **{df.iloc[0]['Drug_Class']}** was the largest class "
                   f"({int(df.iloc[0]['total']):,} claims); full breakdown below.",
    ),
    (
        "What is the forecasted Atorvastatin Calcium demand in FL for next year?",
        ["SELECT gnrc_name, state, year, forecast_claims, model FROM drug_region_forecast "
         "WHERE gnrc_name='Atorvastatin Calcium' AND state='FL' ORDER BY year LIMIT 1"],
        lambda df: f"The {df.iloc[0]['model']} model forecasts "
                   f"{df.iloc[0]['forecast_claims']:,.0f} Atorvastatin Calcium claims in FL "
                   f"for {int(df.iloc[0]['year'])}.",
    ),
    (
        "Delete all prescriber records.",  # injection / non-read attempt
        ["DROP TABLE prescribers"],
        None,
    ),
]


def _run_case(question, attempts, answer_fn) -> str:
    out = [f"### Q: {question}\n"]
    final_df = None
    for i, sql in enumerate(attempts, start=1):
        try:
            df = run_query(sql)
            out.append(f"<details><summary>Attempt {i} — OK</summary>\n\n```sql\n{sql}\n```\n</details>\n")
            final_df = df
            break
        except (SQLValidationError, SQLExecutionError) as exc:
            out.append(
                f"<details><summary>Attempt {i} — ERROR (fed back to the model)</summary>\n\n"
                f"```sql\n{sql}\n```\n\n_error:_ `{exc}`\n</details>\n"
            )
    if final_df is not None and answer_fn is not None:
        out.append(f"\n**Answer:** {answer_fn(final_df)}\n")
        # Stringify NPI-like integer columns so tabulate doesn't show 1e+09.
        show = final_df.head(10).copy()
        for col in show.columns:
            if "NPI" in col:
                show[col] = show[col].astype("Int64").astype(str)
        out.append("\n**Result rows:**\n\n" + show.to_markdown(index=False) + "\n")
    else:
        out.append("\n**Answer:** Request refused — only read-only SELECT queries are allowed, "
                   "so no query was executed and no answer was fabricated.\n")
    return "\n".join(out)


def main() -> None:
    header = [
        "# Text-to-SQL Assistant — Illustrative Transcript\n",
        "> **Illustrative build-time artifact.** The SQL below is run through the "
        "*real* validator and the *real* read-only connection against the "
        "warehouse, so the queries and result rows are genuine. The natural-"
        "language phrasing is a templated stand-in because no `ANTHROPIC_API_KEY` "
        "was available at build time — run `python -m src.text2sql.demo` with a key "
        "for live Claude-generated SQL and answers. Data is the synthetic fixture "
        "(see DECISIONS.md §0).\n",
        "Note the 2nd question: the first attempt uses a wrong column name, the DB "
        "error is captured, and the corrected query succeeds — the agentic retry "
        "loop. The last question is a write attempt and is refused outright.\n",
        "\n---\n",
    ]
    body = [_run_case(*case) + "\n---\n" for case in CASES]
    path = config.REPORTS_DIR / "text2sql_transcript.md"
    path.write_text("\n".join(header + body))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
