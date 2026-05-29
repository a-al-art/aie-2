import argparse
from src.models.train import train


def main():
    parser = argparse.ArgumentParser(description="Обучение модели кредитного скоринга")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Путь к файлу конфигурации",
    )
    args = parser.parse_args()
    train(config_path=args.config)


if __name__ == "__main__":
    main()
