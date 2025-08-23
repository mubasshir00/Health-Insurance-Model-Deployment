from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "ml-inference"
    ENV: str = "dev"
    HMAC_SECRET: str = "change-me"
    
    class Config:
        env_file = ".env"


settings = Settings()