from pydantic import BaseModel
class ApiResponse(BaseModel):
    ok: bool
    message: str | None = None
