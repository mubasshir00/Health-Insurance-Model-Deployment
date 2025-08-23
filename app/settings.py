from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "ml-inference"
    ENV: str = "prod"
    MODEL_PATH: str = "./model.onnx"
    HMAC_SECRET: str = "change-me"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()
