from pydantic_settings import BaseSettings, SettingsConfigDict
from sophons.agents import Agent
from sophons.integrations.models import DeepSeekModel
from sophons.observability import SophonsTelemetry
from sophons.tools import tool


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    otel_exporter_otlp_endpoint: str | None = None


settings = Settings()


telemetry = SophonsTelemetry()
telemetry.setup_console_exporter()
if settings.otel_exporter_otlp_endpoint:
    telemetry.setup_otlp_exporter(settings.otel_exporter_otlp_endpoint)


@tool
def add(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product."""
    return a * b


def main() -> None:
    agent = Agent(
        model=DeepSeekModel(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
        ),
        tools=[add, multiply],
        system_prompt="Always use the provided tools for arithmetic.",
    )
    result = agent.run_sync("What is (1234 + 5678) * 3?", session_id="traced-demo")
    print(result.message)


if __name__ == "__main__":
    main()
