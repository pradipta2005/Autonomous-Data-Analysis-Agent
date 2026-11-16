import time
from typing import Any, Dict, List, Optional
import pandas as pd

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

import utils
from utils import ChartCollector, SafePandasREPL


def build_system_prompt(df: pd.DataFrame) -> str:
    """Construct schema grounding and behavioral guidelines for the analysis agent."""
    profile = utils.generate_data_profile(df)
    columns_info = "\n".join([f"- `{col}` ({df[col].dtype})" for col in df.columns])
    sample_preview = df.head(3).to_string()

    numeric_list = ", ".join([f"`{c}`" for c in profile["numeric_cols"]]) or "None"
    cat_list = ", ".join([f"`{c}`" for c in profile["categorical_cols"]]) or "None"
    date_list = ", ".join([f"`{c}`" for c in profile["date_candidates"]]) or "None"

    return f"""You are an expert Data Analyst and Business Intelligence Assistant.
You have access to a clean pandas DataFrame named `df` loaded in an in-memory execution environment.

### Dataset Overview:
- Records: {profile['rows']:,} rows, {profile['columns']} columns
- Metrics (numeric): {numeric_list}
- Dimensions (categorical): {cat_list}
- Date fields: {date_list}
- Completeness: {profile['completeness_pct']}% populated ({profile['missing_cells']:,} missing cells)
- Duplicates: {profile['duplicate_count']}

### First 3 Sample Rows:
```
{sample_preview}
```

### Analysis & Reporting Guidelines:
1. **Accurate & Computed Numbers**:
   - Always run Python code to compute aggregates, statistics, and percentages before quoting them in your answer.
   - Never approximate or invent values. Format dollar amounts cleanly (e.g. `$12,450.00`) and percentages (e.g. `18.5%`).

2. **Structured Findings**:
   - Present your final conclusion with clear markdown sections:
     - **Executive Summary**: 1–2 sentences summarizing the core finding.
     - **Key Metrics**: Bulleted concrete metrics derived directly from your code output.
     - **Insights & Patterns**: Notable observations, breakdowns by category/region, or anomalies.
     - **Recommendations**: Practical, data-backed next steps for decision makers.
     - **Verification**: A brief closing note confirming the record count and attributes evaluated.

3. **Interactive Charts**:
   - When a visual is helpful, create a Plotly figure and assign it to `fig` (e.g. `fig = px.bar(...)` or `fig = px.line(...)`).
   - Add clean titles, axis labels, and data labels (e.g. `text_auto='.2s'`).
   - Do NOT call `fig.show()`. Assigning to `fig` will automatically render the chart in the UI.

4. **Self-Correction & Safety**:
   - If a `KeyError` or syntax issue arises, inspect the error output, verify column names against the schema above, and re-execute cleanly.
   - OS, disk writing, and network operations are restricted by security policy.
"""


class DataAnalysisAgentManager:
    """Manages the LangChain agent execution loop, prompt binding, and conversational memory."""

    def __init__(
        self,
        df: pd.DataFrame,
        api_key: str,
        model_name: str = "gemini-3.1-flash-lite",
        temperature: float = 0.0,
    ):
        self.df = df
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

        self.chart_collector = ChartCollector()
        self.repl = SafePandasREPL(df=self.df, chart_collector=self.chart_collector)

        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=self.temperature,
        )
        self.agent_executor = self._build_agent_executor()

    def _build_agent_executor(self) -> AgentExecutor:
        repl = self.repl

        @tool
        def execute_pandas_code(code: str) -> str:
            """
            Execute Python/Pandas code in an isolated REPL with pre-loaded `df`, `pd`, `np`, `px`, and `go`.
            Assign Plotly charts to `fig` to display them in the interface. Do not call fig.show().
            """
            return repl.execute(code)

        tools = [execute_pandas_code]
        prompt = ChatPromptTemplate.from_messages([
            ("system", build_system_prompt(self.df)),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            return_intermediate_steps=True,
            max_iterations=5,
            handle_parsing_errors=True,
        )

    def run(self, query: str, chat_history_records: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Execute a user query through the agentic loop, collecting outputs, steps, and visual figures."""
        start_time = time.time()
        self.chart_collector.clear()
        self.repl.reset_namespace()

        # Build recent message history (last 3 conversational turns)
        history = []
        if chat_history_records:
            for msg in chat_history_records[-6:]:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    history.append(HumanMessage(content=content))
                elif role == "assistant":
                    history.append(AIMessage(content=content))

        try:
            result = self.agent_executor.invoke({
                "input": query,
                "chat_history": history,
            })

            raw_output = result.get("output", "Analysis completed.")
            if isinstance(raw_output, list):
                text_parts = [
                    item.get("text", str(item)) if isinstance(item, dict) else str(item)
                    for item in raw_output
                ]
                output_text = "\n\n".join(text_parts)
            elif isinstance(raw_output, dict):
                output_text = raw_output.get("text", str(raw_output))
            else:
                output_text = str(raw_output)

            elapsed = round(time.time() - start_time, 2)

            return {
                "output": output_text,
                "intermediate_steps": result.get("intermediate_steps", []),
                "figures": list(self.chart_collector.get_figures()),
                "execution_time": elapsed,
                "error": None,
            }

        except Exception as exc:
            elapsed = round(time.time() - start_time, 2)
            return {
                "output": f"An unexpected error occurred during analysis: {exc}",
                "intermediate_steps": [],
                "figures": [],
                "execution_time": elapsed,
                "error": str(exc),
            }
