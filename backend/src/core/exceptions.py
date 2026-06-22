"""
全局异常定义模块
"""
from typing import Optional, Any


class BaseAppException(Exception):
    """应用基础异常类"""
    
    def __init__(
        self,
        message: str,
        code: int = 500,
        details: Optional[Any] = None
    ):
        self.message = message
        self.code = code
        self.details = details
        super().__init__(self.message)


class TrainingException(BaseAppException):
    """训练相关异常"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, code=5001, details=details)


class GeneticAlgorithmException(BaseAppException):
    """遗传算法相关异常"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, code=5002, details=details)


class ModelException(BaseAppException):
    """模型相关异常"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, code=5003, details=details)


class ValidationException(BaseAppException):
    """验证相关异常"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, code=4001, details=details)


class ResourceNotFoundException(BaseAppException):
    """资源未找到异常"""
    
    def __init__(self, resource: str, identifier: Any):
        message = f"Resource '{resource}' with identifier '{identifier}' not found"
        super().__init__(message, code=4004, details={"resource": resource, "identifier": identifier})


class PipelineException(BaseAppException):
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, code=5005, details=details)


class ConfigurationException(BaseAppException):
    """配置相关异常"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message, code=5004, details=details)
