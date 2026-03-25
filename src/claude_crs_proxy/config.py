from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    crs_base_url: str = Field(default="https://cc.yintian.vip")
    request_timeout_seconds: float = Field(default=120.0)
    forward_unknown_fields: bool = Field(default=True)
    enable_model_remap: bool = Field(default=True)
    log_request_body: bool = Field(default=False)
    big_model: str = Field(default="gpt-5-codex")
    small_model: str = Field(default="gpt-5.1-codex-mini")

    model_config = SettingsConfigDict(
        env_prefix="CLAUDE_CRS_PROXY_",
        env_file=".env",
        extra="ignore",
    )

    @property
    def openai_base_url(self) -> str:
        return f"{self.crs_base_url.rstrip('/')}" + "/openai"


settings = Settings()
