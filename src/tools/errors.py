# src/tools/errors.py
class ToolError(Exception):
    def __init__(self, code: str, message: str, hint: str = ""):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint

    def to_dict(self):
        return dict(ok=False, error=dict(code=self.code, message=self.message, hint=self.hint))
