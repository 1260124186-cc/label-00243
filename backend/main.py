"""
应用主入口
Non-Differentiable Network Reinforcement Learning System
"""
import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

# 添加src到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import settings
from src.core.logging import setup_logging
from src.core.exceptions import BaseAppException
from src.api.routes import router


# 设置日志
setup_logging(
    log_level=settings.LOG_LEVEL,
    log_file=settings.LOG_FILE
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    # 创建必要目录
    os.makedirs(settings.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    yield
    
    logger.info("Shutting down application")


# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    description="""
## 不可微神经网络强化学习系统

本系统实现了在LunarLander-v2环境中使用PPO算法训练不可微神经网络的可微版本，
并使用遗传算法搜索最优种子组合以生成网络权重。

### 主要功能

1. **PPO训练**: 训练可微网络，通过温度退火逐渐逼近不可微版本
2. **遗传算法搜索**: 搜索最优的24个整数种子组合
3. **模型评估**: 评估训练好的模型性能，判断是否达到200分及格线
4. **对比分析**: 比较可微和不可微网络的性能差异

### 网络架构

- 输入: 8维状态向量 (LunarLander-v2)
- 第一层: Attention-based selection layer
- 第二层: 同结构
- 输出层: Linear -> 4维动作空间

### 技术栈

- Python 3.11+
- FastAPI
- PyTorch
- Gymnasium
    """,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 全局异常处理
@app.exception_handler(BaseAppException)
async def app_exception_handler(request: Request, exc: BaseAppException):
    """处理应用自定义异常"""
    logger.error(f"Application error: {exc.message}", extra={"details": exc.details})
    return JSONResponse(
        status_code=200,  # 业务错误返回200，通过code区分
        content={
            "code": exc.code,
            "message": exc.message,
            "data": exc.details,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """处理未捕获的异常"""
    logger.exception(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "code": 500,
            "message": "Internal server error",
            "data": str(exc) if settings.DEBUG else None,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
    )


# 注册路由
app.include_router(router)

# 挂载plots静态文件目录（file_url可访问）
_plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(_plots_dir, exist_ok=True)
app.mount("/plots", StaticFiles(directory=_plots_dir), name="plots")


# 根路径
@app.get("/", tags=["Root"])
async def root():
    """根路径，返回API信息"""
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
