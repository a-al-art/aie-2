import uvicorn
from src.utils import load_config


def main():
    cfg = load_config("configs/config.yaml")
    svc = cfg.get("service", {})
    uvicorn.run(
        "src.service.app:app",
        host=svc.get("host", "0.0.0.0"),
        port=svc.get("port", 8000),
        log_level=svc.get("log_level", "info"),
        reload=False,
    )


if __name__ == "__main__":
    main()