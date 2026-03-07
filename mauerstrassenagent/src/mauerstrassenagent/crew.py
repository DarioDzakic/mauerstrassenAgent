import os

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.mcp import MCPServerStdio
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from pydantic import BaseModel


# ── Structured output ──
class PortfolioRecommendation(BaseModel):
    stocks: list[dict]  # {ticker, trend, allocation_pct, shares, rationale}
    summary: str
    total_capital: float
    risk_level: str


# ── Knowledge sources (from knowledge/ directory) ──
strategy_knowledge = TextFileKnowledgeSource(
    file_paths=[
        "knowledge/investment_strategies.md",
        "knowledge/sector_analysis.md",
        "knowledge/risk_frameworks.md",
    ]
)


@CrewBase
class MauerstrassenAgent:
    """MauerstrassenAgent — AI Financial Manager Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # ── Agents ──
    @agent
    def strategy_advisor(self) -> Agent:
        return Agent(
            config=self.agents_config["strategy_advisor"],  # type: ignore[index]
            knowledge_sources=[strategy_knowledge],
        )

    @agent
    def market_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["market_analyst"],  # type: ignore[index]
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
        )

    @agent
    def stock_screener(self) -> Agent:
        return Agent(
            config=self.agents_config["stock_screener"],  # type: ignore[index]
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
        )

    @agent
    def report_checker(self) -> Agent:
        return Agent(
            config=self.agents_config["report_checker"],  # type: ignore[index]
            tools=[SerperDevTool()],  # fallback
            mcps=[
                MCPServerStdio(
                    command="python",
                    args=["mcp_server/server.py"],
                    env={"FINNHUB_API_KEY": os.getenv("FINNHUB_API_KEY", "")},
                ),
            ],
        )

    @agent
    def portfolio_builder(self) -> Agent:
        return Agent(
            config=self.agents_config["portfolio_builder"],  # type: ignore[index]
        )

    # ── Tasks ──
    @task
    def strategy_task(self) -> Task:
        return Task(config=self.tasks_config["strategy_task"])  # type: ignore[index]

    @task
    def trend_task(self) -> Task:
        return Task(
            config=self.tasks_config["trend_task"],  # type: ignore[index]
            context=[self.strategy_task()],
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
            context=[
                self.strategy_task(),
                self.screening_task(),
                self.checker_task(),
            ],
            output_pydantic=PortfolioRecommendation,
        )

    # ── Crew ──
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,  # type: ignore[arg-type]
            tasks=self.tasks,  # type: ignore[arg-type]
            process=Process.hierarchical,
            manager_llm="gpt-4o",
            verbose=True,
        )
