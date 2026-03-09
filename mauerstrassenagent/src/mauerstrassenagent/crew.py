from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from pydantic import BaseModel
from typing import List
import os

from mauerstrassenagent.mcp.mcp_servers import get_mcp_tools

from dotenv import load_dotenv
load_dotenv()


# --- Structured output ---
class PortfolioRecommendation(BaseModel):
    stocks: list[dict]  # {ticker, trend, allocation_pct, shares, rationale}
    summary: str
    total_capital: float
    risk_level: str


# --- Knowledge sources ---
strategy_source = TextFileKnowledgeSource(
    file_paths=[
        "investment_strategies.md",
        "risk_frameworks.md",
    ]
)
sector_source = TextFileKnowledgeSource(
    file_paths=["sector_analysis.md"]
)

# --- Shared tools ---
search_tool = SerperDevTool()


@CrewBase
class MauerstrassenAgent:
    """MauerstrassenAgent — AI Financial Manager Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    mcp_tools = get_mcp_tools()

    # --- Agents (defined in sequential execution order) ---
    @agent
    def strategy_advisor(self) -> Agent:
        return Agent(
            config=self.agents_config['strategy_advisor'],  # type: ignore[index]
            verbose=True,
            knowledge_sources=[strategy_source, sector_source],
        )

    @agent
    def market_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['market_analyst'],  # type: ignore[index]
            verbose=True,
            tools=[search_tool],
        )

    @agent
    def stock_screener(self) -> Agent:
        return Agent(
            config=self.agents_config['stock_screener'],  # type: ignore[index]
            verbose=True,
            tools=[search_tool],
        )

    @agent
    def stock_checker(self) -> Agent:
        return Agent(
            config=self.agents_config['stock_checker'],  # type: ignore[index]
            verbose=True,
            tools=self.mcp_tools + [search_tool],
        )

    @agent
    def portfolio_builder(self) -> Agent:
        return Agent(
            config=self.agents_config['portfolio_builder'],  # type: ignore[index]
            verbose=True,
            knowledge_sources=[strategy_source],
        )

    # ── Tasks (strategy → trends → screening → check → portfolio) ──
    @task
    def strategy_task(self) -> Task:
        return Task(
            config=self.tasks_config["strategy_task"],  # type: ignore[index]
            human_input=True
        )

    @task
    def trend_task(self) -> Task:
        return Task(
            config=self.tasks_config["trend_task"],  # type: ignore[index]
            context=[self.strategy_task()],
            human_input=True
        )

    @task
    def screening_task(self) -> Task:
        return Task(
            config=self.tasks_config["screening_task"],  # type: ignore[index]
            context=[self.strategy_task(), self.trend_task()],
        )

    @task
    def checker_task(self) -> Task:
        return Task(
            config=self.tasks_config["checker_task"],  # type: ignore[index]
            context=[self.screening_task()],
        )

    @task
    def portfolio_task(self) -> Task:
        return Task(
            config=self.tasks_config["portfolio_task"],  # type: ignore[index]
            context=[self.strategy_task(), self.checker_task()],
            markdown=True,
            output_file='output/portfolio_recommendation.md'
        )

    # ── Crew ──
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,  # type: ignore[arg-type]
            tasks=self.tasks,  # type: ignore[arg-type]
            process=Process.hierarchical,
            manager_llm=os.getenv("MANAGER_MODEL"),
            verbose=True,
            embedder={  # type: ignore[arg-type]
                "provider": "google-generativeai",
                "config": {
                    "model_name": os.getenv("EMBEDDINGS_GOOGLE_GENERATIVE_AI_MODEL_NAME"),
                    "api_key": os.getenv("GEMINI_API_KEY"),
                },
            },
        )
