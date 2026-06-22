from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    frontend_url: str = "http://localhost:3000"
    frontend_url2: str = "http://localhost:3001"
    extra_allowed_origins: str = (
        "https://frontend-teal-five-45.vercel.app,"
        "https://frontend-gzihc2rum-jeshwanth-as-projects.vercel.app"
    )
    groq_model: str = "llama-3.3-70b-versatile"
    maxi: int = 200
    embedder_model: str = "all-MiniLM-L6-v2"
    GROQ_API_KEY: str

    @property
    def allowed_origins(self) -> list[str]:
        origins = [
            origin.strip().rstrip("/")
            for origin in (
                self.frontend_url,
                self.frontend_url2,
                *self.extra_allowed_origins.split(","),
            )
            if origin and origin.strip()
        ]
        return list(dict.fromkeys(origins))

    class Config:
        env_file = ".env"

settings = Settings()



