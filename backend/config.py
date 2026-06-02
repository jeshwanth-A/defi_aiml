from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    frontend_url: str = "http://localhost:3000"
    frontend_url2 : str = "http://localhost:3001"
    groq_model : str = "llama-3.3-70b-versatile"
    maxi : int = 200
    embedder_model : str = "all-MiniLM-L6-v2"
    GROQ_API_KEY : str

    class Config:
        env_file = ".env"

settings = Settings()



