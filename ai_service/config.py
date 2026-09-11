from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    internal_api_key: str
    google_api_key: str
    openrouter_api_key: str
    cohere_api_key: str = ''
    cloudinary_url: str = ''
    openrouter_chat_models: str = 'google/gemini-3.1-flash-lite,openai/gpt-5.4-mini'

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')


settings = Settings()
